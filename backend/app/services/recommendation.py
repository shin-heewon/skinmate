from typing import List
from sqlalchemy.orm import Session
from app.repository.recommendation import RecommendationRepository
from app.services.diagnosis import DiagnosisService
from app.core.config.logging import get_logger
from app.ai.recommendation.process import expand_query, filter_by_price, generate_recommendations
from app.ai.recommendation.retriever import get_retriever

logger = get_logger(__name__)

class RecommendationService:
    
    @staticmethod
    def create_recommendations(
        db: Session, 
        analysis_id: int,
    ) -> List:
        # 순환 import 방지를 위해 함수 내부에서 import
        from app.services.analysis import AnalysisService
        
        # 1. 진단 정보 조회
        diagnosis = DiagnosisService.get_by_analysis_id(db, analysis_id)
        
        if not diagnosis or not diagnosis.disease_name or not diagnosis.summary:
            raise ValueError(f"진단 정보가 없습니다. analysis_id: {analysis_id}")
        
        # 2. 분석 정보 조회 (필터용)
        analysis = AnalysisService.get_by_id(db, analysis_id)
        
        # 3. Query Expansion (LLM 기반 질의 확장)
        keyword_query, semantic_query = expand_query(
            disease_name=diagnosis.disease_name,
            summary=diagnosis.summary,
            skin_type=analysis.skin_type if analysis else None
        )
        
        # 4. Hybrid Retrieval (EnsembleRetriever)
        retriever = get_retriever()
        
        # search_kwargs에 filter 전달 (pre-filter)
        search_kwargs = {}
        if analysis and analysis.skin_type:
            search_kwargs["filter"] = {"skin_type": analysis.skin_type}
            logger.info(f"Pre-filter 적용: skin_type={analysis.skin_type}")
        
        search_results = retriever.get_relevant_documents(
            semantic_query,
            **search_kwargs if search_kwargs else {}
        )
        logger.info(f"Pre-filter 후 검색 결과: {len(search_results)}개")
        
        # 5. Post-filter (가격 필터링)
        if analysis:
            min_price = float(analysis.min_price) if analysis.min_price is not None else None
            max_price = float(analysis.max_price) if analysis.max_price is not None else None
            logger.info(f"가격 필터 적용: min_price={min_price}, max_price={max_price}")
            
            before_price_filter = len(search_results)
            search_results = filter_by_price(
                search_results,
                min_price=min_price,
                max_price=max_price
            )
            after_price_filter = len(search_results)
            logger.info(f"가격 필터 적용 후: {before_price_filter}개 -> {after_price_filter}개")
            
            # 가격 필터로 결과가 3개 미만이면 가격 필터 완화
            if after_price_filter < 3 and before_price_filter > 0:
                if after_price_filter == 0:
                    logger.warning("가격 필터로 결과가 0개가 되었습니다. 가격 필터 없이 재검색합니다.")
                else:
                    logger.warning(f"가격 필터로 결과가 {after_price_filter}개로 3개 미만입니다. 가격 필터를 완화합니다.")
                # 필터 없이 재검색
                search_results = retriever.get_relevant_documents(semantic_query)
                logger.info(f"가격 필터 제거 후 검색 결과: {len(search_results)}개")
        
        # 상위 10개만 선택
        search_results = search_results[:10]
        logger.info(f"최종 검색 결과: {len(search_results)}개")
        
        if not search_results:
            logger.warning("검색 결과가 없습니다.")
            return []
        
        # 6. LLM Recommendation (LLM 2)
        recommendation_query = f"{diagnosis.disease_name} 피부질환 진단에 대한 화장품을 추천해줘. 진단 요약: {diagnosis.summary}"
        
        recommendations = generate_recommendations(
            query=recommendation_query,
            search_results=search_results
        )
        
        # 7. DB에 저장 (LLM이 반환한 결과를 그대로 사용)
        recommendations_data = [
            {
                "analysis_id": analysis_id,
                "cosmetic_id": rec["cosmetic_id"],
                "ranking": rec["ranking"],
                "reason": rec["reason"]
            }
            for rec in recommendations[:3]
        ]
        
        return RecommendationRepository.create_bulk(db, recommendations_data)
