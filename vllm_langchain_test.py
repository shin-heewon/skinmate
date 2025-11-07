# vLLM + Langchain을 사용한 모델 테스트
# vLLM 배포 후 실행 (포트 8000)
# 
# 사용 방법:
# 1. Step 3에서 vLLM 배포 완료
# 2. pip install -U langchain-openai
# 3. python vllm_langchain_test.py

from vllm.multimodal.utils import encode_image_base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image

# Langchain ChatOpenAI 설정
llm = ChatOpenAI(
    model="./model_16bit",  # vLLM 배포 시 사용한 모델명
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1"  # vLLM 배포 포트
)

# 피부 질환 진단 프롬프트
instruction = """너는 안면부 피부 질환을 분석하는 전문 AI이다. 
주어진 얼굴 부위 피부 이미지를 관찰하고, 이미지에서 보이는 임상적 특징을 자세히 설명하라.

**중요 지침:**
- 다음 피부 질환 목록 중 가장 두드러진 주된 질환 1개를 <label>에 명시하라
- summary에서 동반 가능한 다른 질환의 소견이 있다면 함께 언급할 수 있다
- 3문장 이내로 간결하면서도 핵심적인 정보를 담아라. 같은 표현 반복을 피하라
- 과도한 추측보다는 이미지에서 관찰 가능한 객관적 소견 및 특징에 근거하여 기술하라

다음은 진단 가능한 피부 질환 목록과 각 질환의 임상적 특징이다:

**0: 건선 (Psoriasis)**
- 병변 형태: 은백색 인설이 쌓인 붉은 구진이나 판
- 경계: 매우 명확하고 뚜렷함
- 안면 발생: 이마, 헤어라인, 귀 주변에서 관찰 가능
- 핵심 특징: 두꺼운 은백색 인설, 명확한 경계, 대칭적 분포
- 증상: 가려움증 동반 가능

**1: 아토피 피부염 (Atopic Dermatitis)**
- 병변 형태: 건조하고 가려운 습진성 병변, 태선화
- 경계: 불명확
- 안면 발생: 얼굴 전반, 특히 뺨, 이마, 눈 주위
- 핵심 특징: 피부 건조, 긁은 자국, 만성 재발성
- 증상: 심한 가려움증

**2: 여드름 (Acne)**
- 병변 형태: 면포(comedone), 구진, 농포, 낭종
- 경계: 개별 병변은 명확
- 안면 발생: 이마, 코, 턱 등 T존 중심, 뺨에도 가능
- 핵심 특징: 다양한 병변 동시 존재, 피지선 분포 부위
- 증상: 염증성 병변은 통증 가능

**3: 주사 (Rosacea)**
- 병변 형태: 지속적인 홍반, 모세혈관 확장, 구진, 농포
- 경계: 불명확한 홍반
- 안면 발생: 얼굴 중앙부(코, 뺨 중심, 이마)
- 핵심 특징: 안면 홍조, 혈관 확장 두드러짐, 딸기코 가능
- 증상: 작열감, 따끔거림

**4: 지루 피부염 (Seborrheic Dermatitis)**
- 병변 형태: 기름기 있는 노란 비늘과 홍반
- 경계: 비교적 명확
- 안면 발생: 눈썹, 비구순 주름, 귀 주변, 헤어라인
- 핵심 특징: 기름진 각질, 피지선이 많은 부위
- 증상: 가려움증, 각질

**5: 정상 (Normal)**
- 특징: 특별한 병변이 관찰되지 않음
- 피부 상태: 건강한 피부 톤과 질감

---

**답변 형식:**
<label>{질병명}</label>
<summary>{이미지에서 관찰되는 구체적 소견을 자세히 기술. 병변의 색상, 형태, 경계, 분포, 크기 등을 포함하여 해당 질환의 특징적 소견임을 설명.}</summary>

**예시 1:**
<label>건선</label>
<summary>이미지에서는 이마와 헤어라인 부위에 홍반성 판이 관찰되며, 그 위로 은백색의 두꺼운 인설이 층을 이루어 쌓여있습니다. 병변의 경계가 매우 명확하여 주변 정상 피부와 뚜렷하게 구분됩니다. 인설의 두께와 은백색 광택, 명확한 경계는 건선의 전형적인 임상 양상입니다.</summary>

**예시 2:**
<label>여드름</label>
<summary>이미지에서는 얼굴의 이마와 뺨 부위에 다수의 홍반성 구진과 농포가 관찰됩니다. 일부 병변은 중심부에 화농성 내용물이 있으며, 면포도 함께 보입니다. 피지선이 발달한 안면부에 염증성 병변과 비염증성 병변이 혼재된 양상은 심상성 여드름의 특징적 소견입니다.</summary>

**예시 3:**
<label>정상</label>
<summary>이미지에서는 특별한 병변이나 이상 소견이 관찰되지 않습니다. 피부 톤이 균일하고 질감이 매끄러우며, 홍반, 구진, 인설 등의 병적 변화가 없습니다. 건강한 정상 피부 상태를 보이고 있습니다.</summary>
"""

# 테스트 이미지 로드 및 인코딩
# 본인의 테스트 이미지 경로로 변경하세요
image_path = "./test_image.png"

try:
    image = Image.open(image_path).convert("RGB")
    image_base64 = encode_image_base64(image)
    
    # Langchain 메시지 생성 (이미지 + 텍스트)
    # 주의: Image와 Text를 동시에 넣는 것은 ChatCompletion 형태여야 함
    # TextCompletion으로는 실행되지 않음
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        )
    ]
    
    # 모델 호출
    print("=" * 60)
    print("vLLM + Langchain 테스트 시작")
    print("=" * 60)
    print(f"이미지: {image_path}")
    print("\n진단 결과:")
    print("-" * 60)
    
    response = llm.invoke(messages)
    print(response.content)
    
    print("-" * 60)
    print("테스트 완료!")
    
except FileNotFoundError:
    print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
    print("image_path 변수를 본인의 테스트 이미지 경로로 변경하세요.")
except Exception as e:
    print(f"오류 발생: {str(e)}")
    print("\n확인 사항:")
    print("1. vLLM이 포트 8000에서 실행 중인지 확인")
    print("2. langchain-openai가 설치되어 있는지 확인: pip install -U langchain-openai")
    print("3. 이미지 경로가 올바른지 확인")

