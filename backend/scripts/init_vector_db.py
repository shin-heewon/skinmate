"""
1. 벡터 DB 최초 적재 스크립트 - 화장품 데이터를 Qdrant에 업서트
2. 사용법 (모듈 실행):
    python -m scripts.init_vector_db
"""
import sys
import os
from sqlalchemy.orm import Session
from langchain_core.documents import Document
from app.core.config.database import get_db
from app.core.config.logging import get_logger
from app.core.config.vectordb import (
    init_cosmetics_collection, QDRANT_URL, QDRANT_API_KEY, 
    OPENAI_API_KEY, COLLECTION_NAME
)
from app.models.cosmetic import Cosmetic
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# 프로젝트 루트(backend/)를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = get_logger(__name__)


def get_cosmetic_documents(db: Session) -> list[Document]:
    cosmetics = db.query(Cosmetic).all()
    
    documents = []
    for c in cosmetics:
        # page_content 생성
        page_content = c.description
        
        # metadata 생성 (필터)
        metadata = {
            "cosmetic_id": c.cosmetic_id,
            "skin_type": c.skin_type or "",
            "price": float(c.price) if c.price else 0.0,
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    logger.info(f"화장품 데이터 변환 완료: {len(documents)}개")
    return documents


def upsert_cosmetics(db: Session) -> None:
    if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY:
        raise ValueError("QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY가 설정되지 않았습니다.")
    
    # 1. Document 로드
    documents = get_cosmetic_documents(db)
    if not documents:
        logger.warning("업서트할 화장품 데이터가 없습니다.")
        return
    
    # 2. Embeddings 생성
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # 3. Qdrant에 업서트
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    logger.info(f"업서트 완료: {len(documents)}개 화장품")


def main():
    """벡터 DB 최초 적재 스크립트"""
    logger.info("벡터 DB 초기화 시작")
    
    db: Session = next(get_db())
    
    try:
        # 1. 컬렉션 확인/생성
        init_cosmetics_collection()
        
        # 2. 데이터 업서트
        upsert_cosmetics(db)
        
        logger.info("벡터 DB 초기화 완료")
    except Exception as e:
        logger.error(f"벡터 DB 초기화 실패: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()

