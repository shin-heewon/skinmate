from sqlalchemy.orm import Session
from fastapi import UploadFile, status
from typing import Optional
from app.repository.analysis import AnalysisRepository
from app.repository.analysis_view import AnalysisViewRepository
from app.services.file import FileService
from app.services.diagnosis import DiagnosisService
from app.services.recommendation import RecommendationService
from app.schemas.analysis import AnalysisResponse, AnalysisHistoryResponse, AnalysisHistoryItem
from app.schemas.recommendation import Recommendation as RecommendationSchema
from app.core.exception import ApiException
from app.core.config.file import get_static_file_url


class AnalysisService:
    
    @staticmethod
    def create_analysis(
        db: Session,
        member_id: int,

        image_file: UploadFile, #필수값

        skin_type: Optional[str] = None, # 옵셔널
        min_price: Optional[int] = None, # 옵셔널
        max_price: Optional[int] = None, # 옵셔널
    ) -> int:
        
        # 1. skin_analysis 생성 (옵셔널 데이터)
        analysis = AnalysisRepository.create(db, {
            "member_id": member_id,
            "skin_type": skin_type or None,
            "min_price": min_price or None,
            "max_price": max_price or None,
            "created_id": member_id,
        })
        analysis_id = analysis.analysis_id
        
        # 2. 파일 업로드 (FileService 호출)
        FileService.upload_and_save(db, analysis_id, image_file)
        
        # 3. 진단 생성 (파인튜닝 모델 호출)
        DiagnosisService.create_diagnosis(db, analysis_id)
        
        # 4. 추천 생성 (더미 데이터 -> 추후 RAG 파이프라인 구축)
        RecommendationService.create_recommendations(db, analysis_id)
        
        # 5. analysis_id만 반환
        return analysis_id
    
    
    @staticmethod
    def get_analysis_result(db: Session, analysis_id: int) -> AnalysisResponse:
        
        # 1. DB View에서 데이터 조회 (ranking 순 정렬됨)
        view_results = AnalysisViewRepository.get_by_analysis_id(db, analysis_id)
        
        if not view_results:
            raise ApiException(status.HTTP_404_NOT_FOUND, "분석 결과를 찾을 수 없습니다")
        
        # 2. 첫 번째 row에서 기본 정보 추출
        first_row = view_results[0]
        
        # 3. 추천 제품 리스트 구성 (cosmetic_id가 있는 row만)
        recommendation_list = []
        for row in view_results:
            if row.cosmetic_id:  # cosmetic이 존재하는 경우만
                recommendation_list.append(RecommendationSchema(
                    name=row.cosmetic_name or "",
                    brand=row.brand or "",
                    price=float(row.price) if row.price else 0,
                    file_path=get_static_file_url(row.cosmetic_file_path) if row.cosmetic_file_path else None,
                    buy_url=row.buy_url or "",
                    reason=row.reason or ""
                ))
        
        # 4. 응답 반환
        return AnalysisResponse(
            analysis_id=first_row.analysis_id,
            file_id=first_row.skin_file_id or 0, # 유저가 업로드한 피부 이미지파일 ID
            disease_name=first_row.disease_name or "",
            diagnosis_summary=first_row.diagnosis_summary or "",
            products=recommendation_list,
            created_at=first_row.analysis_created_at
        )
    
    
    @staticmethod
    def get_analysis_history(db: Session, member_id: int, page: int = 1, size: int = 10, 
                           disease_name: str = None, period: str = "all") -> AnalysisHistoryResponse:
        """
        분석 이력 목록 조회
        
        Args:
            db: 데이터베이스 세션
            member_id: 회원 ID
            page: 페이지 번호 (기본값: 1)
            size: 페이지 크기 (기본값: 10)
            disease_name: 진단명 필터링 (선택적)
            period: 기간 필터링 (all/day/week/month, 기본값: all)
            
        Returns:
            AnalysisHistoryResponse: 페이징된 이력 목록
        """
        # 1. 페이징된 이력 조회
        history_items = AnalysisRepository.get_by_member_id_with_pagination(
            db, member_id, page, size, disease_name, period
        )
        
        # 2. 전체 개수 조회 (필터링 조건 반영)
        total = AnalysisRepository.count_by_member_id(db, member_id, disease_name, period)
        
        # 3. AnalysisHistoryItem 리스트 생성
        items = []
        for item in history_items:
            items.append(AnalysisHistoryItem(
                analysis_id=item.analysis_id,
                disease_name=item.disease_name or "",
                created_at=item.created_at
            ))
        
        # 4. 응답 반환
        return AnalysisHistoryResponse(
            items=items,
            total=total,
            page=page,
            size=size
        )
    
    
    @staticmethod
    def delete_analysis(db: Session, analysis_id: int, member_id: int) -> bool:
        """
        분석 이력 삭제
        
        Args:
            db: 데이터베이스 세션
            analysis_id: 분석 ID
            member_id: 회원 ID (권한 검증용)
            
        Returns:
            bool: 삭제 성공 여부
        """
        # 1. 분석 이력 존재 확인
        analysis = AnalysisRepository.get_by_id(db, analysis_id)
        if not analysis:
            raise ApiException(status.HTTP_404_NOT_FOUND, "분석 이력을 찾을 수 없습니다")
        
        # 2. 권한 검증: 본인의 데이터만 삭제 가능
        if analysis.member_id != member_id:
            raise ApiException(status.HTTP_403_FORBIDDEN, "본인의 분석 이력만 삭제할 수 있습니다")
        
        # 3. 삭제 실행
        success = AnalysisRepository.delete_by_id(db, analysis_id)
        
        if not success:
            raise ApiException(status.HTTP_500_INTERNAL_SERVER_ERROR, "분석 이력 삭제에 실패했습니다")
        
        return True
    
    @staticmethod
    def get_by_id(db: Session, analysis_id: int):
        return AnalysisRepository.get_by_id(db, analysis_id)

