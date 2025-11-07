# -*- coding: utf-8 -*-
"""
Step 2: GPT-4로 피부 상태 분석 생성

- 이미 파싱된 데이터셋에 GPT-4 설명 추가
- 순수 GPT 출력만 저장 (후처리 없음)
- 중단 후 재실행 가능!

⚠️ 주의: 이 단계에서는 후처리를 하지 않습니다.
         step3_check_issues.py로 문제점 확인 후
         step4_fix_dataset.py에서 후처리를 적용합니다.
"""

import base64
from io import BytesIO
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 시스템 프롬프트 (프롬프트 수정: _정면/_측면 제거)
# ============================================================

SYSTEM_PROMPT = """You are an expert AI for analyzing facial skin diseases.
Observe the given facial skin image and describe the clinical features visible in the image in detail.

**Important Guidelines:**
- Specify one primary disease that is most prominent from the disease list below in <label>
- In summary, you may mention findings of other possible concurrent diseases if observed
- Base your description on objective findings and features observable in the image rather than excessive speculation
- **Provide detailed and comprehensive descriptions (minimum 3-5 sentences, at least 150 characters)**
- Include specific observations about color, morphology, borders, distribution, size, and texture
- **Frontal images**: Observe the entire face to analyze overall skin condition and distribution patterns of lesions
- **Side images**: Since these are magnified images of specific areas, carefully observe detailed lesion morphology, borders, color, texture, etc.

The following is a list of diagnosable skin diseases and their characteristics:

**Psoriasis (건선)**
- Lesion morphology: Red papules or plaques with thick accumulation of silvery-white scales
- Border: Very clear and distinct (clearly differentiated from surrounding normal skin)
- Facial distribution: Commonly occurs on forehead, hairline, eyebrows, around ears
- Features: Dry and rough skin, silvery-white lustrous scales, tendency for symmetric distribution
- Observation points: Thickness of scales, degree of erythema, clarity of lesion borders
- Possible accompanying symptoms: Skin cracking, bleeding in severe cases

**Atopic Dermatitis (아토피)**
- Lesion morphology: Dry and red eczematous lesions, scratch marks, scale lifting
- Border: Unclear and diffuse appearance
- Facial distribution: Common on cheeks, forehead, chin, neck areas
- Features: Extreme dryness, skin cracking, lichenification (in chronic cases), scratch marks
- Observation points: Degree of dryness, presence of scratch marks, extent of erythema, scale lifting
- Possible accompanying symptoms: Exudate, discharge, signs of secondary infection

**Acne (여드름)**
- Lesion morphology: Blackheads/whiteheads (comedones), red papules, pus-filled pustules, cysts
- Border: Individual lesions are clear, multiple lesions distributed
- Facial distribution: Especially numerous in T-zone (forehead, nose, chin), possible on cheeks
- Features: Enlarged pores, excessive sebum, mixed stages of lesions, coexistence of inflammatory/non-inflammatory lesions
- Observation points: Number of comedones, degree of inflammation, presence of pustules, scars or marks
- Possible accompanying symptoms: Acne marks (PIE/PIH), scars, enlarged pores

**Rosacea (주사)**
- Lesion morphology: Persistent erythema, telangiectasia (visible blood vessels), occasionally papules/pustules
- Border: Unclear and diffuse erythema
- Facial distribution: Central face - symmetric on cheeks, nose, forehead, chin
- Features: Red skin, visible blood vessels, easily flushed, can be confused with acne
- Observation points: Persistence of flushing, degree of telangiectasia, erythema distribution pattern
- Possible accompanying symptoms: Facial edema, stinging, burning sensation, acne-like lesions

**Seborrheic Dermatitis (지루)**
- Lesion morphology: Greasy yellowish scales and erythema
- Border: Relatively clear
- Facial distribution: Scalp borderline, eyebrows, around nose, nasolabial folds, around ears, T-zone
- Features: Greasy and moist scales, excessive sebum, yellowish scales, erythema
- Observation points: Greasiness of scales, degree of yellowness, T-zone oiliness, scale appearance
- Possible accompanying symptoms: Itching, scalp dandruff, concurrent facial and scalp seborrhea

**Normal (정상)**
- Lesions: No particular lesions or abnormal findings
- Border: None
- Skin condition: Uniform skin tone, healthy texture
- Features: Good moisture-oil balance, normal pores, elastic
- Observation points: Uniformity of skin tone, smooth skin texture, no erythema/inflammation, no scales
- Overall features: Healthy and stable skin condition

---

**Answer Format (IMPORTANT: Answer in Korean):**
<label>질병명</label>
<summary>이미지에서 관찰되는 구체적 소견을 자세히 기술. 병변의 색상, 형태, 경계, 분포, 크기 등을 포함하여 해당 질환의 특징적 소견임을 설명.</summary>

**Example 1:**
<label>건선</label>
<summary>이미지에서는 팔꿈치 부위에 홍반성 판이 관찰되며, 그 위로 은백색의 두꺼운 인설이 층을 이루어 쌓여있습니다. 병변의 경계가 매우 명확하여 주변 정상 피부와 뚜렷하게 구분됩니다. 인설의 두께와 은백색 광택, 명확한 경계는 건선의 전형적인 임상 양상입니다.</summary>

**Example 2:**
<label>여드름</label>
<summary>이미지에서는 얼굴의 이마와 뺨 부위에 다수의 홍반성 구진과 농포가 관찰됩니다. 일부 병변은 중심부에 화농성 내용물이 있으며, 면포도 함께 보입니다. 피지선이 발달한 안면부에 염증성 병변과 비염증성 병변이 혼재된 양상은 심상성 여드름의 특징적 소견입니다. 추가로 뺨 부위에 약한 홍조와 모세혈관 확장이 관찰되어 주사가 동반되어 있을 가능성도 있습니다.</summary>

**Example 3:**
<label>정상</label>
<summary>이미지에서는 특별한 병변이나 이상 소견이 관찰되지 않습니다. 피부 톤이 균일하고 질감이 매끄러우며, 홍반, 구진, 인설 등의 병적 변화가 없습니다. 건강한 정상 피부 상태를 보이고 있습니다.</summary>

---

**Special Cases:**
- If the image is not of a facial area, or if image quality is too poor to make a judgment:
  <label>진단불가</label>
  <summary>제공된 이미지는 얼굴 부위가 아니거나 / 이미지 품질이 불량하여 피부 질환을 판단할 수 없습니다.</summary>
"""

# ============================================================
# GPT-4로 설명 생성
# ============================================================

def process_dataset_image(dataset, split, index, llm):
    """GPT-4로 이미지 분석 (순수 출력, 후처리 없음)"""
    if 'image' not in dataset[split].features:
        return "The dataset does not contain an 'image' column."

    pil_image = dataset[split][index]['image']
    label = dataset[split][index]['label']
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    sys_message = SystemMessage(content=SYSTEM_PROMPT)
    
    # label은 step1에서 이미 정리됨 (건선, 아토피 등)
    message = HumanMessage(content=[
        {"type": "text", "text": f"이 환자는 {label}로 진단되었다. 이미지에서 관찰되는 {label}의 특징적 소견을 자세히 설명해라."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
    ])

    chain = llm | StrOutputParser()
    response = chain.invoke([sys_message, message])
    
    # 후처리 없음: GPT 원본 출력 그대로 반환
    return response


def main():
    import os
    import shutil
    
    # 로컬에 저장할 경로
    SAVE_PATH = "./skin_dataset"
    TEMP_PATH = "./skin_dataset_temp"
    
    # 환경변수에서 API 키 읽기 (.env 파일 사용)
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일 로드
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        # 환경변수에 없으면 직접 입력
        api_key = input("OpenAI API Key를 입력하세요: ").strip()
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=api_key
    )
    
    print("=" * 60)
    print("Step 2: GPT-4로 피부 상태 분석 생성 (통합 버전)")
    print("=" * 60)
    
    dataset = load_from_disk(SAVE_PATH)
    print("\n데이터셋 로드 완료!")
    print(dataset)
    
    # ========== Train 처리 ==========
    print("\n" + "=" * 60)
    print("Train 데이터 처리 중...")
    print("=" * 60)
    
    # 사전 작업: output 컬럼이 없으면 빈 문자열로 초기화
    if "output" not in dataset["train"].features:
        dataset["train"] = dataset["train"].add_column("output", [""] * len(dataset["train"]))

    # 기존 output을 리스트로 복사
    outputs = dataset["train"]["output"][:]
    
    # 진행 루프
    for i in tqdm(range(len(dataset["train"])), desc="Train", mininterval=1.0):
        if outputs[i]:  # 이미 처리된 경우 skip
            continue
        try:
            result = process_dataset_image(dataset, "train", i, llm)
        except Exception as e:
            print(f"\nERROR at train index {i}: {str(e)}")
            result = ""  # 빈 문자열로 저장, 나중에 재처리 가능

        outputs[i] = result
        
        # 150개마다 중간 저장 (에러 대비)
        if (i + 1) % 150 == 0 or i == len(dataset["train"]) - 1:
            dataset["train"] = dataset["train"].remove_columns("output").add_column("output", outputs)
            
            # 임시 경로에 저장 후 이동 (파일 핸들 충돌 방지)
            if os.path.exists(TEMP_PATH):
                shutil.rmtree(TEMP_PATH)
            dataset.save_to_disk(TEMP_PATH)
            
            del dataset  # 메모리에서 해제
            if os.path.exists(SAVE_PATH):
                shutil.rmtree(SAVE_PATH)
            shutil.move(TEMP_PATH, SAVE_PATH)
            
            print(f"\n[저장 완료: {i+1}/{len(outputs)}]")
            
            # 재로드
            dataset = load_from_disk(SAVE_PATH)

    # ========== Test 처리 ==========
    print("\n" + "=" * 60)
    print("Test 데이터 처리 중...")
    print("=" * 60)
    
    # 사전 작업: output 컬럼이 없으면 빈 문자열로 초기화
    if "output" not in dataset["test"].features:
        dataset["test"] = dataset["test"].add_column("output", [""] * len(dataset["test"]))

    # 기존 output을 리스트로 복사
    outputs_test = dataset["test"]["output"][:]
    
    # 진행 루프
    for i in tqdm(range(len(dataset["test"])), desc="Test", mininterval=1.0):
        if outputs_test[i]:  # 이미 처리된 경우 skip
            continue
        try:
            result = process_dataset_image(dataset, "test", i, llm)
        except Exception as e:
            print(f"\nERROR at test index {i}: {str(e)}")
            result = ""  # 빈 문자열로 저장, 나중에 재처리 가능

        outputs_test[i] = result
        
        # 150개마다 중간 저장 (에러 대비)
        if (i + 1) % 150 == 0 or i == len(dataset["test"]) - 1:
            dataset["test"] = dataset["test"].remove_columns("output").add_column("output", outputs_test)
            
            # 임시 경로에 저장 후 이동 (파일 핸들 충돌 방지)
            if os.path.exists(TEMP_PATH):
                shutil.rmtree(TEMP_PATH)
            dataset.save_to_disk(TEMP_PATH)
            
            del dataset  # 메모리에서 해제
            if os.path.exists(SAVE_PATH):
                shutil.rmtree(SAVE_PATH)
            shutil.move(TEMP_PATH, SAVE_PATH)
            
            print(f"\n[저장 완료: {i+1}/{len(outputs_test)}]")
            
            # 재로드
            dataset = load_from_disk(SAVE_PATH)
    
    print("\n" + "=" * 60)
    print("처리 완료!")
    print("=" * 60)
    print(f"\n저장 위치: {SAVE_PATH}")
    print("=" * 60)
    print("\n✅ GPT 설명 생성이 완료되었습니다!")
    print("=" * 60)
    print("\n다음 단계:")
    print("  1. python pipeline/step3_check_issues.py - 문제점 자동 탐지")
    print("  2. config_postprocess.py 수정 - 발견된 문제 설정")
    print("  3. python pipeline/step4_fix_dataset.py - 후처리 적용")
    print("=" * 60)


if __name__ == "__main__":
    main()

