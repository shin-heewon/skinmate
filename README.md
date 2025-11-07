# 🏥 피부 질환 데이터셋 전처리 가이드

피부 질환 진단을 위한 Vision-Language 모델 학습용 데이터셋 구축 파이프라인입니다.

---

## ⚠️ 작업 환경 준비 (필수)

**중요**: 이 데이터셋 전처리 작업은 **별도의 작업 폴더**에서 진행하는 것을 강력히 권장합니다.

### 왜 별도 폴더가 필요한가요?

데이터 전처리 과정에서 다음과 같은 대용량 파일들이 생성됩니다:
- 📁 `extracted_data/` - 원본 이미지 (10,800장, 수 GB)
- 📁 `skin_dataset/` - 중간 데이터셋 (수 GB)
- 📁 `skin_dataset_fixed/` - 최종 데이터셋 (수 GB)
- 📁 `skin_dataset_temp/` - 임시 파일

**메인 프로젝트 폴더에서 작업하면**:
- ❌ 프로젝트가 매우 지저분해짐
- ❌ Git 관리가 복잡해짐 (용량 큰 파일 추적)
- ❌ 불필요한 파일로 인한 혼란

### ✅ 권장 작업 방법

#### 방법 1: 별도 프로젝트 폴더 생성 (추천)

```bash
# 데이터셋 전처리 전용 폴더 생성
mkdir skin-dataset-build
cd skin-dataset-build

# 이 브랜치 클론
git clone -b docs/dataset-preprocessing https://github.com/your-username/skin-diagnosis.git .

# 이미지 다운로드 및 배치
# extracted_data/ 폴더에 AI Hub 이미지 배치

# 작업 진행
python pipeline/step1_parse_dataset.py
python pipeline/step2_add_descriptions.py
python pipeline/step3_check_issues.py
# config_postprocess.py 수정
python pipeline/step4_fix_dataset.py
python pipeline/step5_upload_to_hub.py
```

#### 방법 2: 메인 프로젝트 외부에 작업 폴더

```
작업 구조:
├── skin-diagnosis/            # 메인 서비스 (Git 관리)
│   ├── api/
│   ├── rag/
│   └── ...
│
└── dataset-workspace/         # 데이터셋 작업 (별도 폴더)
    ├── pipeline/              # 이 브랜치 클론
    ├── config_postprocess.py
    ├── extracted_data/        # 원본 이미지
    ├── skin_dataset/          # 중간 파일
    └── skin_dataset_fixed/    # 최종본 → HuggingFace 업로드
```

### 작업 완료 후

최종 데이터셋(`skin_dataset_fixed/`)만:
- ✅ HuggingFace Hub에 업로드 (`step5_upload_to_hub.py`)
- ✅ 또는 필요시 메인 프로젝트에서 다운로드

**중간 파일들은 로컬에만 보관하거나 삭제하세요.**

---

## 📊 데이터셋 정보

- **대상 질환**: 건선, 아토피, 여드름, 주사, 지루, 정상 (6개 클래스)
- **데이터 규모**: Train 9,600개 / Test 1,200개
- **이미지 해상도**: 1024x1024 PNG
- **설명 생성**: GPT-4o-mini 기반 의학적 분석

---

## 🔄 전처리 워크플로우

```
Step 1: 이미지 파싱 + label 정리
  ↓
./skin_dataset/ (label 정리됨, output 비어있음)
  ↓
Step 2: GPT-4 설명 생성 (순수 출력)
  ↓
./skin_dataset/ (GPT 원본 출력)
  ↓
Step 3: 문제점 자동 탐지
  ↓
리포트 확인 → config_postprocess.py 수정
  ↓
Step 4: 후처리 적용
  ↓
./skin_dataset_fixed/ ✅ 최종본
  ↓
Step 5: HuggingFace Hub 업로드 (선택)
  ↓
🌐 HuggingFace Hub
```

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp env.example .env
# .env 파일을 열어서 OpenAI API 키를 입력하세요
```

### 2️⃣ 데이터 준비

원본 이미지를 다음 구조로 배치하세요:

**데이터 출처**: [AI Hub - 안면부 피부질환 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71863)

```
skin_test/
├── extracted_data/
│   ├── Training/
│   │   └── images/
│   │       ├── TS_건선_정면/    (800장)
│   │       ├── TS_건선_측면/    (800장)
│   │       ├── TS_아토피_정면/  (800장)
│   │       ├── TS_아토피_측면/  (800장)
│   │       ├── TS_여드름_정면/  (800장)
│   │       ├── TS_여드름_측면/  (800장)
│   │       ├── TS_주사_정면/    (800장)
│   │       ├── TS_주사_측면/    (800장)
│   │       ├── TS_지루_정면/    (800장)
│   │       ├── TS_지루_측면/    (800장)
│   │       ├── TS_정상_정면/    (800장)
│   │       └── TS_정상_측면/    (800장)
│   └── Validation/
│       └── images/
│           ├── VS_건선_정면/    (100장)
│           ├── VS_건선_측면/    (100장)
│           ├── VS_아토피_정면/  (100장)
│           ├── VS_아토피_측면/  (100장)
│           ├── VS_여드름_정면/  (100장)
│           ├── VS_여드름_측면/  (100장)
│           ├── VS_주사_정면/    (100장)
│           ├── VS_주사_측면/    (100장)
│           ├── VS_지루_정면/    (100장)
│           ├── VS_지루_측면/    (100장)
│           ├── VS_정상_정면/    (100장)
│           └── VS_정상_측면/    (100장)
```

**⚠️ 중요**: 
- 폴더명은 정확히 위와 같아야 합니다
- 이미지 파일은 `.png`, `.jpg`, `.jpeg` 형식 지원
- 각 폴더에 정확한 수량의 이미지가 있어야 합니다

---

## 📝 5단계 실행 가이드

### Step 1: 이미지 파싱 + label 정리

```bash
python pipeline/step1_parse_dataset.py
```

**기능**:
- `extracted_data/` 폴더의 이미지 재귀 탐색
- HuggingFace Dataset 형식으로 변환
- **자동 label 정리**: `건선_정면` → `건선`

**출력**: `./skin_dataset/` (초기 데이터셋)

---

### Step 2: GPT-4 설명 생성

```bash
python pipeline/step2_add_descriptions.py
```

**기능**:
- GPT-4o-mini로 각 이미지의 의학적 분석 생성
- **순수 GPT 출력만 저장** (후처리 없음)
- 자동 중간 저장 (150개 단위)

**출력**: `./skin_dataset/` (GPT 원본 출력)

**⚠️ 중요**: 
- **이 단계는 매우 오래 걸립니다** (10,800회 API 호출)
- 중간 저장(150개 단위)이 자동으로 되므로, 중단되어도 이어서 실행 가능
- 이 단계에서는 영어 번역이나 라벨 통일을 하지 않습니다
- GPT의 원본 출력을 그대로 저장합니다

**💡 최적화 옵션**: 이미지 해상도 축소
- 현재 코드는 1024x1024 해상도 사용
- 512x512로 축소하면 **비용/시간 상당히 절감 가능** (이미지 토큰만 기준)
- 실제 절감율은 프롬프트 길이, 응답 길이에 따라 달라질 수 있음
- 장단점을 고려하여 선택 (아래 "이미지 해상도 최적화" 섹션 참고)

---

### Step 3: 문제점 자동 탐지

```bash
python pipeline/step3_check_issues.py
```

**기능**:
- 영어 단어 자동 탐지 (빈도 포함)
- 라벨 불일치 체크 (`label` 필드 vs `<label>` 태그)
- 통계 리포트 출력

**출력 예시**:
```
🔤 발견된 영어 단어 (빈도순)
  - erythema                     (125회)
  - papule                       (89회)
  - telangiectasia               (34회)

⚠️  라벨 불일치
  - label='아토피' vs <label>아토피 피부염</label> (120건)
  - label='지루' vs <label>지루성 피부염</label> (85건)

💡 권장 사항
1️⃣  config_postprocess.py의 MEDICAL_TERMS에 추가:
    "erythema": "홍반",
    "papule": "구진",
    
2️⃣  config_postprocess.py의 LABEL_MAPPING에 추가:
    '아토피 피부염': '아토피',
    '지루성 피부염': '지루',
```

**다음 작업**:
1. 리포트를 확인
2. `config_postprocess.py` 파일을 열기
3. 발견된 영어 용어를 `MEDICAL_TERMS`에 추가
4. 발견된 라벨 변형을 `LABEL_MAPPING`에 추가

---

### Step 4: 후처리 적용

**⚠️ 작업 전 필수**: `config_postprocess.py`를 수정하여 Step 3에서 발견된 문제점 반영

**중요**: 
- `config_postprocess.py`의 `MEDICAL_TERMS`와 `LABEL_MAPPING`은 **예시**입니다
- **반드시 Step 3의 리포트 결과를 확인**하고 실제 발견된 항목으로 수정하세요
- GPT 출력은 매번 다를 수 있으므로, 실제 데이터에 맞춰 조정이 필요합니다

```bash
python pipeline/step4_fix_dataset.py
```

**기능**:
- `config_postprocess.py`의 설정을 로드
- 영어 의학 용어 → 한글 번역
- output의 `<label>` 태그 통일

**출력**: `./skin_dataset_fixed/` (최종 데이터셋)

**처리 과정 예시**:
```python
# config_postprocess.py에서 로드 (Step 3 결과에 맞춰 수정)
MEDICAL_TERMS = {
    "erythema": "홍반",        # Step 3에서 발견된 경우 추가
    "papule": "구진",          # Step 3에서 발견된 경우 추가
    # Step 3 리포트의 영어 단어를 모두 추가
}

LABEL_MAPPING = {
    '아토피 피부염': '아토피',  # Step 3에서 발견된 경우 추가
    '지루성 피부염': '지루',    # Step 3에서 발견된 경우 추가
    # Step 3 리포트의 라벨 변형을 모두 추가
}

# 자동 적용
"erythema and papule" → "홍반 and 구진"
"<label>아토피 피부염</label>" → "<label>아토피</label>"
```

**검증 (선택)**:
```bash
# 문제가 해결되었는지 재확인
python pipeline/step3_check_issues.py
```

---

### Step 5: HuggingFace Hub 업로드 (선택)

```bash
python pipeline/step5_upload_to_hub.py
```

**기능**:
- `./skin_dataset_fixed/` → HuggingFace Hub 업로드
- `.env`에서 `HF_TOKEN` 읽기
- 메타데이터 자동 생성

**필수 설정** (`.env` 파일):
```
HF_TOKEN=your_huggingface_token
HF_REPO_NAME=username/dataset-name
```

---

## 📁 파일 구조

### 실행 스크립트 (pipeline/)

| 파일                         | 설명                      | 입력                      | 출력                       |
|------------------------------|---------------------------|---------------------------|--------------------------|
| `step1_parse_dataset.py`     | 이미지 파싱 + label 정리   | `extracted_data/`         | `./skin_dataset/`         |
| `step2_add_descriptions.py`  | GPT-4 설명 생성            | `./skin_dataset/`         | `./skin_dataset/`        |
| `step3_check_issues.py`      | 문제점 자동 탐지           | `./skin_dataset/`         | 리포트 출력                 |
| `step4_fix_dataset.py`       | 후처리 적용                | `./skin_dataset/`         | `./skin_dataset_fixed/`  |
| `step5_upload_to_hub.py`     | HuggingFace Hub 업로드     | `./skin_dataset_fixed/`   | Hub                      |

### 설정 파일

| 파일                     | 설명                                                       |
|--------------------------|-----------------------------------------------------------|
| `config_postprocess.py`  | 영어 용어 + 라벨 매핑 (⚠️ 예시, Step 3 결과에 맞춰 수정)       |
| `env.example`            | 환경변수 템플릿                                             |
| `requirements.txt`       | 필요한 패키지 목록                                          |

---

## 📊 최종 데이터 통계

| 클래스   | Train     | Test      | 합계      |
|----------|-----------|-----------|-----------|
| 건선     | 1,600     | 200       | 1,800     |
| 아토피   | 1,600     | 200       | 1,800     |
| 여드름   | 1,600     | 200       | 1,800     |
| 주사     | 1,600     | 200       | 1,800     |
| 지루     | 1,600     | 200       | 1,800     |
| 정상     | 1,600     | 200       | 1,800     |
| **합계** | **9,600** | **1,200** | **10,800** |

---

## 🔧 문제 해결

### Q1. Step 2에서 중단되었어요
**A**: `step2_add_descriptions.py`를 다시 실행하세요. 이미 처리된 데이터는 자동으로 건너뜁니다.

### Q2. Step 3에서 영어 단어가 많이 발견되었어요
**A**: 정상입니다. `config_postprocess.py`에 해당 용어를 추가한 후 Step 4를 실행하세요.

### Q3. config_postprocess.py를 어떻게 수정하나요?
**A**: 
```python
# Step 3 리포트 확인
🔤 발견된 영어 단어
  - erythema (125회)
  - papule (89회)

# config_postprocess.py의 MEDICAL_TERMS에 추가
MEDICAL_TERMS = {
    "erythema": "홍반",    # Step 3에서 발견됨
    "papule": "구진",      # Step 3에서 발견됨
    # 기존 예시는 참고용, 실제 발견된 단어만 추가하면 됨
}
```

**주의**: config_postprocess.py에 이미 작성된 내용은 예시입니다. Step 3 결과를 보고 실제로 필요한 항목만 추가/수정하세요.

### Q4. Step 4 후에도 문제가 남아있어요
**A**: 
1. `python pipeline/step3_check_issues.py` 재실행
2. 새로 발견된 항목을 `config_postprocess.py`에 추가
3. `python pipeline/step4_fix_dataset.py` 재실행

### Q5. API 키 오류가 발생해요
**A**: `.env` 파일에 올바른 OpenAI API 키가 입력되었는지 확인하세요.

---

## 🌐 프롬프트 언어 선택 (영어 vs 한글)

현재 `step2_add_descriptions.py`의 시스템 프롬프트는 **영어**로 작성되어 있습니다.

### 📝 현재 구현

```python
SYSTEM_PROMPT = """You are an expert AI for analyzing facial skin diseases.
Observe the given facial skin image and describe the clinical features...

**Answer Format (IMPORTANT: Answer in Korean):**
<label>질병명</label>
<summary>이미지에서 관찰되는 구체적 소견...</summary>
"""
```

- **프롬프트**: 영어
- **요구 출력**: 한글 (`Answer in Korean` 명시)

### 💡 영어 프롬프트를 사용하는 이유

**토큰 절약**: 영어가 한글보다 토큰 효율이 좋음
```
영어: "Observe the facial skin image" → 6 토큰
한글: "얼굴 피부 이미지를 관찰하세요" → 15+ 토큰
```

복잡한 의학적 지시사항이 많을수록 토큰 차이가 커집니다.

### ⚠️ 중요: 영어 의학 용어 발생에 대한 오해

**프롬프트가 영어라서 영어 단어가 나오는 것이 아닙니다:**

- ✅ 프롬프트를 **한글로 변경해도** 영어 의학 용어는 여전히 나올 수 있음
- ✅ `Answer in Korean` 지시가 있어도 GPT가 가끔 영어 의학 용어 사용
- ✅ 이는 GPT 학습 데이터에 영어 의학 문헌이 많기 때문

**실제 경험**:
- 한글 프롬프트 사용 시에도 "erythema", "papule" 등 영어 용어 발생
- 프롬프트 언어와 무관하게 Step 3-4의 후처리가 필요함

### 📊 프롬프트 언어별 장단점

| 구분               | 영어 프롬프트                    | 한글 프롬프트                      |
|--------------------|--------------------------------|----------------------------------|
| **토큰 효율**       | ✅ 더 적은 토큰 사용             | ❌ 더 많은 토큰 사용               |
| **모델 성능**       | ✅ GPT가 영어에 최적화됨         | ⚠️ 약간 낮을 수 있음               |
| **가독성**          | ❌ 한국 개발자가 수정하기 어려움  | ✅ 쉽게 이해하고 수정 가능          |
| **영어 단어 출현**   | ⚠️ 가끔 나옴                    | ⚠️ 역시 가끔 나옴 (거의 동일)       |

### 💡 권장 사항

**프로토타이핑/비용 중시**: 영어 프롬프트 (현재 구현)
- 토큰 절약으로 비용 감소
- GPT 성능 최대 활용

**프롬프트 수정 빈도 높음**: 한글 프롬프트
- 의학적 기준이나 예시를 자주 수정해야 한다면
- 팀 협업 시 가독성 중요

**어느 쪽을 선택하든**:
- Step 3-4의 후처리 파이프라인은 필수
- 영어 의학 용어는 두 경우 모두 발생 가능

### 🔧 한글 프롬프트로 변경하려면

`step2_add_descriptions.py`의 `SYSTEM_PROMPT`를 한글로 번역:

```python
SYSTEM_PROMPT = """당신은 안면 피부 질환 분석 전문 AI입니다.
주어진 얼굴 피부 이미지를 관찰하고 이미지에서 보이는 임상적 특징을 상세히 설명하세요.

**중요 지침:**
- 아래 질환 목록에서 가장 두드러진 주요 질환 하나를 <label>에 명시하세요
- 요약에서는 관찰된 다른 동반 가능 질환의 소견도 언급할 수 있습니다
...
"""
```

---

## ⚡ 이미지 해상도 최적화 (선택사항)

현재 파이프라인은 1024x1024 해상도를 사용하지만, 비용과 시간을 절감하기 위해 해상도를 축소할 수 있습니다.

### 📐 해상도별 비교

| 해상도           | 이미지 토큰  | 장점                          | 단점                      |
|-----------------|------------|-------------------------------|---------------------------|
| **1024x1024**   | ~765 토큰   | 세밀한 병변 관찰<br>정확한 분석   | 비용/시간 많이 소요         |
| **512x512**     | ~255 토큰   | 비용 절감<br>처리 시간 단축      | 미세 특징 손실<br>품질 저하  |

**⚠️ 주의**: 
- 위 토큰 수는 이미지 자체만 해당
- 실제 API 비용은 시스템 프롬프트 + 사용자 메시지 + 응답 토큰도 포함
- 따라서 실제 절감율은 예상보다 낮을 수 있음

### ✅ 512x512로 축소 고려할 경우

- 💰 **비용 절감**을 원할 때 (단, 큰 폭 절감은 보장 안 됨)
- 🚀 **빠른 프로토타이핑**이 필요할 때
- 📊 **전체적인 패턴/분포**만 중요할 때
- 🔍 병변이 **크고 명확**한 경우

### ⚠️ 1024x1024를 유지해야 하는 경우

- 🏥 **의학적 정확도**가 최우선일 때
- 🔬 **세밀한 병변 분석**이 필요할 때 (인설 두께, 모세혈관 확장 등)
- 📝 작은 구진, 면포 등 **미세 병변** 관찰이 중요할 때
- ✨ **최종 프로덕션** 데이터셋 제작 시

### 🔧 구현 방법

`step2_add_descriptions.py`의 `process_dataset_image` 함수를 수정:

```python
def process_dataset_image(dataset, split, index, llm):
    # ... (기존 코드)
    
    pil_image = dataset[split][index]['image']
    
    # ✨ 해상도 축소 (원하는 경우 주석 해제)
    # TARGET_SIZE = 512
    # if pil_image.size != (TARGET_SIZE, TARGET_SIZE):
    #     from PIL import Image
    #     pil_image = pil_image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    # ... (나머지 코드)
```

### 💡 권장 전략

**단계적 접근**:
1. 소규모 샘플(100개)로 512x512 테스트
2. 생성된 설명의 품질 확인
3. 만족스러우면 전체 적용, 아니면 1024x1024 유지

**하이브리드 접근**:
- 정면 이미지: 512x512 (전체 패턴 파악)
- 측면 이미지: 1024x1024 (세부 특징 관찰)

---

## 💡 주요 특징

✅ **명확한 단계 분리**: 각 단계가 독립적으로 작동  
✅ **자동 문제 탐지**: 영어 단어, 라벨 불일치 자동 발견  
✅ **유연한 설정**: GPT 출력에 따라 설정 파일만 조정  
✅ **반복 가능**: Step 3-4를 여러 번 반복하여 완벽하게 수정 가능  
✅ **품질 보증**: 최종 검증된 데이터만 HuggingFace에 업로드  
✅ **디버깅 용이**: 각 단계별 중간 결과 확인 가능  
✅ **중단 재개**: Step 2 중단 시 이어서 실행 가능

---

## 📄 라이선스 및 저작권

### 이미지 데이터 저작권

본 프로젝트에서 활용한 **이미지 데이터**는 [AI Hub - 안면부 피부질환 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71863)에서 제공하는 데이터를 사용하였습니다.

**AI Hub 데이터 사용 조건**:
- AI Hub의 데이터 이용 약관을 준수해야 합니다
- 상업적 이용 시 별도의 승인이 필요할 수 있습니다
- 데이터 출처를 반드시 명시해야 합니다

**⚠️ 중요**: 
- AI Hub 데이터를 사용할 경우, [AI Hub 이용약관](https://www.aihub.or.kr/)을 반드시 확인하고 준수하세요
- 이 저장소는 데이터 전처리 **스크립트만** 제공합니다
- 원본 이미지 데이터는 포함되어 있지 않으며, 사용자가 직접 AI Hub에서 다운로드해야 합니다

### 자체 데이터 사용

AI Hub 데이터 대신 **본인이 직접 수집하거나 저작권 문제가 없는 이미지**를 사용할 경우:
- 이 전처리 파이프라인을 자유롭게 사용할 수 있습니다
- 동일한 폴더 구조에 맞춰 이미지를 배치하세요
- 본인 데이터에 대한 저작권 관리는 사용자의 책임입니다

### 코드 라이선스

이 저장소의 **전처리 스크립트 및 코드**는 연구 및 교육 목적으로 자유롭게 사용 가능합니다.


