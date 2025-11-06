from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from app.core.config.database import create_tables
from app.core.config.vectordb import init_rag_system
from app.core.config.file import STATIC_DIR
from app.core.config.cors import get_cors_config
from app.core.config.openapi import custom_openapi
from app.core.exception import ApiException, api_exception_handler
from app.core.middleware.auth_middleware import JWTMiddleware
from app.router import member_router, analysis_router, file_router, like_router, cosmetic_router, test_router, chat_router
from app.core.config.logging import get_logger
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    # 1. RDB 테이블 생성
    create_tables()
    # logger.info("데이터베이스 테이블 생성 완료")
    
    # 2. RAG 시스템 초기화 (Qdrant + Retriever)
    init_rag_system()
    # logger.info("RAG 시스템 초기화 완료")
    
    yield
    
    # 종료 시 실행
    # logger.info("애플리케이션 종료 중")


# FastAPI 애플리케이션 생성 (Swagger 표시 설정)
app = FastAPI(
    title="SkinMate API",
    description="피부질환 진단 및 화장품 추천 서비스 API",
    docs_url="/docs",
    lifespan=lifespan
)

# OpenAPI 스키마 커스터마이징 설정(JWT Bearer 인증 테스트용)
app.openapi = lambda: custom_openapi(app)

# CORS 설정
app.add_middleware(CORSMiddleware, **get_cors_config())

# JWT 검증 미들웨어 등록
app.add_middleware(JWTMiddleware)

# 전역 예외 핸들러 등록
app.add_exception_handler(ApiException, api_exception_handler)

# 라우터 등록
app.include_router(member_router)
app.include_router(analysis_router)
app.include_router(file_router)
app.include_router(like_router)
app.include_router(cosmetic_router)
app.include_router(test_router)
app.include_router(chat_router)

# 정적 파일 서빙 - /media 경로로 마운트 (업로드된 미디어 파일)
app.mount("/media", StaticFiles(directory=STATIC_DIR), name="media")

# 기본 라우트
@app.get("/api")
async def root():
    return {"message": "SkinMate API 서버 실행 성공"}

# 헬스 체크 엔드포인트
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "SkinMate API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True
    )
