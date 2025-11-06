"""Qdrant 클라이언트 및 컬렉션 초기화 관리"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.core.config.logging import get_logger

logger = get_logger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "cosmetics"

# Qdrant 클라이언트 생성 및 반환
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL 또는 QDRANT_API_KEY가 설정되지 않았습니다.")
    
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 컬렉션 존재 여부 확인 및 필요 시 컬렉션 생성
def init_cosmetics_collection(qdrant: QdrantClient = None) -> None:
    if qdrant is None:
        qdrant = get_qdrant_client()
    
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            logger.warning(f"Qdrant 컬렉션 '{COLLECTION_NAME}'이 비어 있습니다. init_vector_db.py를 먼저 실행하세요.")
        else:
            logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME}': {collection_info.points_count}개 포인트")
    
    except Exception:
        # 컬렉션이 존재하지 않으면 컬렉션 생성
        try:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            logger.info("컬렉션 생성 완료")
        except Exception as e:
            logger.warning(f"컬렉션 생성 실패: {e}")

def init_rag_system() -> None:
    """RAG 시스템 초기화 (서버 시작 시 한 번만 실행)"""
    try:
        # 순환 import 방지를 위해 함수 내부에서 import
        from app.ai.recommendation.retriever import init_retriever
        
        # 1. 컬렉션 확인 및 미존재 시 컬렉션 생성
        init_cosmetics_collection()
        
        # 2. Retriever 초기화 (서버 시작 시 한 번만)
        logger.info("서버 시작 시 Retriever 초기화 시작")
        qdrant = get_qdrant_client()
        init_retriever(qdrant)  # init_retriever 내부에서 이미 초기화되어 있으면 재초기화하지 않음
        logger.info("RAG 시스템 초기화 완료 (Qdrant + Retriever)")
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 실패: {e}", exc_info=True)
        raise  # 초기화 실패 시 서버 시작을 중단

