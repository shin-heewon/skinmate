"""Retriever 초기화 및 관리"""
from typing import Optional
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from app.core.config.logging import get_logger
from app.core.config.vectordb import (
    QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, COLLECTION_NAME,
    get_qdrant_client
)

logger = get_logger(__name__)

# 전역 변수: EnsembleRetriever 캐싱
ensemble_retriever: Optional[EnsembleRetriever] = None


def _get_documents_from_qdrant(qdrant_client: QdrantClient) -> list[Document]:
    documents = []
    offset = None
    
    while True:
        # Qdrant에서 모든 포인트를 스크롤하여 가져오기
        result, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not result:
            break
        
        for point in result:
            # payload에서 page_content와 metadata 추출
            payload = point.payload or {}
            page_content = payload.get("page_content", payload.get("description", ""))
            
            if not page_content:
                continue
            
            metadata = {
                "cosmetic_id": payload.get("cosmetic_id", point.id),
                "skin_type": payload.get("skin_type", ""),
                "brand": payload.get("brand", ""),
                "category": payload.get("category", ""),
                "price": payload.get("price", 0.0),
            }
            
            documents.append(Document(page_content=page_content, metadata=metadata))
        
        if offset is None:
            break
    
    logger.info(f"Qdrant에서 {len(documents)}개 문서 로드 완료")
    return documents


def init_retriever(qdrant_client: QdrantClient) -> EnsembleRetriever:
    global ensemble_retriever
    
    if ensemble_retriever is not None:
        logger.info(f"Retriever가 이미 초기화되어 있습니다. (캐시된 인스턴스 재사용)")
        return ensemble_retriever
    
    logger.info("Retriever 초기화 시작 (새로 생성)")
    
    # 1. Qdrant에서 Document 로드
    documents = _get_documents_from_qdrant(qdrant_client)
    if not documents:
        raise ValueError("Qdrant에서 화장품 데이터를 가져올 수 없습니다. init_vector_db.py를 먼저 실행하세요.")
    
    # 2. BM25Retriever 생성
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    logger.info("BM25Retriever 초기화 완료")
    
    # 3. Dense Retriever 생성 (QdrantVectorStore)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # QdrantVectorStore 생성 (langchain-qdrant 패키지 사용)
    qdrant_vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    dense_retriever = qdrant_vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    logger.info("Dense Retriever 초기화 완료")
    
    # 4. EnsembleRetriever 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    logger.info("EnsembleRetriever 초기화 완료")
    
    return ensemble_retriever


def get_retriever() -> EnsembleRetriever:
    """Retriever 인스턴스 반환 (lazy initialization)"""
    global ensemble_retriever
    
    if ensemble_retriever is None:
        logger.info("Retriever가 초기화되지 않았습니다. 지연 초기화를 시작합니다.")
        qdrant = get_qdrant_client()
        init_retriever(qdrant)
    else:
        logger.debug("캐시된 Retriever 반환")
    
    return ensemble_retriever

