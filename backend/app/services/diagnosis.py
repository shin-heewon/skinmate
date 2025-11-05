import re
import os
from typing import Optional
from PIL import Image
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.repository.diagnosis import DiagnosisRepository
from app.repository.file import FileRepository
from app.models.diagnosis import Diagnosis
from app.models.entity_type import EntityType
from app.utils.image import encode_image_base64
from app.utils.prompt import load_prompt
from app.core.exception.exceptions import ApiException
from fastapi import status as http_status

# 프롬프트 로드
INSTRUCTION = load_prompt("diagnosis.yaml")
class DiagnosisService:
    
    @staticmethod
    def create_diagnosis(db: Session, analysis_id: int) -> Diagnosis:
        
        # analysis_id로 파일 조회
        file = FileRepository.get_by_entity(db, EntityType.SKIN_ANALYSIS, analysis_id)

        image = Image.open(file.file_path).convert("RGB")
        image_base64 = encode_image_base64(image)

        llm = ChatOpenAI(
            model=os.getenv("RUNPOD_MODEL_NAME"),
            api_key=os.getenv("RUNPOD_API_KEY"),
            base_url=os.getenv("RUNPOD_BASE_URL"),
            temperature=0.1,
        )

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            )
        ]

        try:
            response = llm.invoke(messages)
            
            # 결과 파싱
            disease = re.search(r"<label>(.*?)</label>", response.content)
            summary = re.search(r"<summary>(.*?)</summary>", response.content)
            
            if not disease or not summary:
                raise ApiException(http_status.HTTP_500_INTERNAL_SERVER_ERROR,"AI 진단 응답 형식이 올바르지 않습니다")
            
            diagnosis_data = {
                "analysis_id": analysis_id,
                "disease_name": disease.group(1),
                "summary": summary.group(1)
            }
            
        except ApiException:
            raise
            
        except Exception as e:
            raise ApiException(
                http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"AI 진단 중 오류가 발생했습니다: {str(e)}"
            )
        
        # 에러 없을 때만 DB에 저장
        return DiagnosisRepository.create(db, diagnosis_data)
    
    @staticmethod
    def get_by_analysis_id(db: Session, analysis_id: int) -> Optional[Diagnosis]:
        return DiagnosisRepository.get_by_analysis_id(db, analysis_id)