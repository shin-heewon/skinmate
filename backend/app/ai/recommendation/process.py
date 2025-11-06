"""RAG 전처리 파이프라인 관리"""
import os
import json
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from app.core.config.logging import get_logger
from app.utils.prompt import load_prompt
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# llm 기반 질의 확장
def expand_query(
    disease_name: str, 
    summary: str, 
    skin_type: Optional[str] = None
) -> Tuple[str, str]:

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    prompt_template = load_prompt("query_expansion.yaml")
    prompt = prompt_template.format(
        disease_name=disease_name,
        summary=summary,
        skin_type=skin_type or "정보 없음"
    )
    
    # JSON 출력 강제
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    result = json.loads(response.content)
    keyword_query = result.get("keyword_query", "")
    semantic_query = result.get("semantic_query", "")
    
    logger.info(
        f"질의 확장 완료 - 입력: disease_name={disease_name}, skin_type={skin_type or '정보 없음'}\n"
        f"  keyword_query: {keyword_query}\n"
        f"  semantic_query: {semantic_query}"
    )
    
    return keyword_query, semantic_query


def filter_by_price(
    results: List[Document],
    min_price: Optional[float],
    max_price: Optional[float]
) -> List[Document]:

    if min_price is None and max_price is None:
        return results
    
    filtered = []
    for doc in results:
        price = doc.metadata.get("price", 0.0)
        
        # 가격 필터링
        if min_price is not None and price < min_price:
            logger.debug(f"가격 필터링 제외: price={price} < min_price={min_price}, cosmetic_id={doc.metadata.get('cosmetic_id')}")
            continue
        if max_price is not None and price > max_price:
            logger.debug(f"가격 필터링 제외: price={price} > max_price={max_price}, cosmetic_id={doc.metadata.get('cosmetic_id')}")
            continue
        filtered.append(doc)
        logger.debug(f"가격 필터링 통과: price={price}, cosmetic_id={doc.metadata.get('cosmetic_id')}")
    
    logger.info(f"가격 필터링 결과: {len(results)}개 -> {len(filtered)}개")
    return filtered


def generate_recommendations(
    query: str,
    search_results: List[Document]
) -> List[Dict]:
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    # context에 각 문서의 cosmetic_id 포함
    context = "\n\n".join([
        f"문서 [cosmetic_id: {doc.metadata.get('cosmetic_id', 'N/A')}]: {doc.page_content}"
        for doc in search_results[:5]
    ])
    
    # 프롬프트 로드
    prompt_template = load_prompt("recommendation_cosmetic.yaml")
    prompt = prompt_template.format(query=query, context=context)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # JSON 출력 강제
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    result = json.loads(response.content)
    recommendations = result.get("recommendations", [])
    
    logger.info(f"LLM 추천 결과: {len(recommendations)}개")
    
    return recommendations

