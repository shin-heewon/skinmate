import os
import re
from sqlalchemy.orm import Session
from app.repository.recommendation import RecommendationRepository
from app.models.cosmetic import Cosmetic
from app.models.diagnosis import Diagnosis
from app.core.config.logging import get_logger
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = get_logger(__name__)

class RecommendationService:
    
    @staticmethod
    def create_recommendations(db: Session, analysis_id: int, member_id: int) -> List:

        # 1. Qdrant Cloud 연결
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        COLLECTION = "cosmetics"

        # 컬렉션 생성 (없으면 새로 생성)
        if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

        # 2. 화장품 메타데이터 → 비정형 텍스트 (DB에서 조회)
        cosmetic_list = db.query(Cosmetic).all()
        
        cosmetics = [
            {
                "id": c.cosmetic_id,
                "name": c.name or "",
                "brand": c.brand or "",
                "category": c.category or "",
                "price": float(c.price) if c.price else 0,
                "ingredients": c.ingredients or "",
                "description": c.description or c.short_description or "",
                "skin_type": c.skin_type or "",
                "skin_disease": c.skin_disease or "",
                "main_effect": c.main_effect or "",
                "care_symptom": c.care_symptom or "",
                "key_ingredient": c.key_ingredient or ""
            }
            for c in cosmetic_list
        ]

        def cosmetic_to_text(c: dict) -> str:
            parts = [
                f"화장품ID: {c['id']}",
                f"제품명: {c['name']}",
                f"브랜드: {c['brand']}",
                f"카테고리: {c['category']}",
                f"가격: {c['price']}원"
            ]
            
            # 있으면 데이터 추가
            if c.get('ingredients'):
                parts.append(f"성분: {c['ingredients']}")
            if c.get('description'):
                parts.append(f"설명: {c['description']}")
            if c.get('skin_type'):
                parts.append(f"피부타입: {c['skin_type']}")
            if c.get('skin_disease'):
                parts.append(f"피부질환: {c['skin_disease']}")
            if c.get('main_effect'):
                parts.append(f"주요효능: {c['main_effect']}")
            if c.get('care_symptom'):
                parts.append(f"케어증상: {c['care_symptom']}")
            if c.get('key_ingredient'):
                parts.append(f"핵심성분: {c['key_ingredient']}")
            
            return "\n".join(parts)

        docs = [cosmetic_to_text(c) for c in cosmetics]
        
        # logger.info(f"First doc preview: {docs[0][:200]}...")

        # 3. Dense Vector
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

        vectorstore = QdrantVectorStore(
            client=qdrant,
            collection_name=COLLECTION,
            embedding=embeddings
        )

        # 데이터 업서트
        for i, c in enumerate(cosmetics):
            vectorstore.add_documents([
                Document(page_content=docs[i], metadata=c)
            ])

        # 4. Sparse Vector (BM25)
        bm25 = BM25Retriever.from_documents([Document(page_content=d) for d in docs])

        # 5. Hybrid Retriever (Dense + Sparse)
        retriever = EnsembleRetriever(
            retrievers=[vectorstore.as_retriever(search_kwargs={"k": 3}), bm25],
            weights=[0.7, 0.3]   # Dense 70%, Sparse 30%
        )

        # 6. 진단 정보 조회 및 질의 생성
        diagnosis = db.query(Diagnosis).filter(Diagnosis.analysis_id == analysis_id).first()
        
        if diagnosis and diagnosis.disease_name and diagnosis.summary:
            query = f"{diagnosis.disease_name} 피부질환에 대한 화장품 추천. 진단 요약: {diagnosis.summary}"
        # else: 예외 처리 추가 구현할것
        #     query = "피부 화장품 추천"
        
        # logger.info(f"🔍 Query: {query}")
        

        # 7. 검색 실행
        results = retriever.get_relevant_documents(query)

        # logger.info("검색된 문서:")
        # for r in results:
        #     logger.info(f"{r.page_content[:200]}...")

        # 8. LLM 프롬프트 + 응답
        client = OpenAI(api_key=OPENAI_API_KEY)

        context = "\n\n".join([r.page_content for r in results])

        prompt = prompt = f"""
        너는 피부 전문가이다.
        사용자 질문: {query}
        참고 문서: {context}

        위 문서를 참고해서 사용자 질문에 답변하되,
        추천 화장품 3개를 아래 형식으로 출력하라.

        형식:
        1. 추천 화장품: [cosmetic_id 숫자], [ranking 숫자], [reason 추천 이유 설명]

        제약 조건:
        - cosmetic_id는 반드시 문서의 metadata에 있는 "id" 값(숫자)만 쓸 것. 제품명이나 브랜드명을 쓰지 말 것.
        - ranking은 '1, 2, 3' 숫자만 출력할 것. '1위', 'Rank 1' 같은 표현 금지.
        - reason은 추천 이유를 설명할 것.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        llm_response = response.choices[0].message.content
        logger.info(f"최종 답변: {llm_response}")

        # 정규표현식으로 파싱 -> 추천 화장품 3개 추출
        pattern = r'\d+\.\s*추천\s*화장품:\s*(\d+),\s*(\d+),\s*(.+?)(?=\d+\.\s*추천\s*화장품:|$)'
        matches = re.findall(pattern, llm_response, re.DOTALL)

        recommendations_data = []
        for cosmetic_id, ranking, reason in matches:
            recommendations_data.append({
                "analysis_id": analysis_id,
                "cosmetic_id": int(cosmetic_id),
                "ranking": int(ranking),
                "reason": reason.strip()
            })
        
        return RecommendationRepository.create_bulk(db, recommendations_data)

