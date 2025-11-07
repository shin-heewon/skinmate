# Vision-Language 모델 파인튜닝 가이드

피부 질환 진단을 위한 Vision-Language 모델 파인튜닝 및 배포 파이프라인입니다.

---

## 목차

1. [RunPod Pod 생성](#1-runpod-pod-생성)
2. [Step 1: 데이터셋 학습](#step-1-데이터셋-학습)
3. [Step 2: 모델 병합](#step-2-모델-병합)
4. [Step 3: vLLM 배포 (중요)](#step-3-vllm-배포-중요)
5. [Step 4: 모델 테스트](#step-4-모델-테스트)

---

## 1. RunPod Pod 생성

### 1-1. 필수 요구사항

**중요**: vLLM 배포를 위해서는 **GPU VRAM 48GB 이상**이 필요합니다.

**권장 GPU**:
- **RTX A6000** (48GB VRAM) - 추천
- **RTX A5000** (24GB VRAM) - 학습용으로만 사용 가능, vLLM 배포는 어려움
- RTX 4090 (24GB VRAM) - vLLM 배포 불가

### 1-2. Pod 생성 단계

1. **RunPod 로그인**: https://www.runpod.io/
2. **"Deploy"** 버튼 클릭
3. **GPU 선택**: 
   - vLLM 배포 목적: **RTX A6000** 권장
   - 학습만 목적: RTX A5000도 가능
4. **템플릿 선택**: **"RunPod Pytorch 2.0"** 또는 **"RunPod Tensorflow"**
5. **Container Disk**: 권장 **50GB** (모델 15GB + 데이터셋 + 캐시)
6. **Volume Disk**: 권장 **100GB** (모델 백업 및 장기 저장용)
7. **"Deploy On-Demand"** 클릭

**디스크 설정 참고**:
- **Container Disk**: 작업 중 임시 파일 저장 (Pod 종료 시 삭제될 수 있음)
- **Volume Disk**: 중요한 모델 백업용 (Pod 종료 후에도 유지)
- Volume Disk는 선택 사항이지만, 모델 백업을 위해 권장합니다

### 1-3. Jupyter Lab 접속

1. Pod 카드에서 **"Connect"** 드롭다운 클릭
2. **"Connect to Jupyter Lab"** 선택
3. 새 브라우저 탭에서 Jupyter Lab 열림

---

## Step 1: 데이터셋 학습

### 중요: 실행 환경

**이 파인튜닝 과정은 RunPod Pod에서 실행해야 합니다.**

- 노트북 파일 3개(`dataset_study.ipynb`, `model_merge.ipynb`, `confusion_matrix.ipynb`)를 RunPod Pod에 복사하여 실행하세요
- 배치 테스트는 `vllm_batch_test.py` 스크립트를 사용합니다 (Step 4 참고)
- 로컬에서 실행 가능하지만, **GPU VRAM이 충분해야 합니다** (최소 24GB 이상 권장)
- 일반적인 로컬 환경에서는 GPU 메모리 부족으로 실행이 어려울 수 있습니다
- RunPod Pod 사용을 강력히 권장합니다

### 1-1. 개인정보 설정

**중요**: 노트북을 실행하기 전에 개인정보를 설정해야 합니다.

#### HuggingFace 토큰 설정

`notebooks/dataset_study.ipynb`의 Cell 4를 수정:

```python
!huggingface-cli login --token your_huggingface_token_here
```

#### WandB API 키 설정

`notebooks/dataset_study.ipynb`의 Cell 14를 수정:

```python
wandb.login(key="your_wandb_api_key_here")
```

#### 데이터셋 이름 설정

`notebooks/dataset_study.ipynb`의 Cell 5를 수정:

```python
from datasets import load_dataset

dataset = load_dataset("your-username/your-dataset-name")
```

### 1-2. 핵심 코드 이해

파인튜닝 학습의 핵심은 두 가지입니다:

#### 1) 데이터셋 변환 (Cell 11)

데이터셋을 Vision-Language 모델이 이해할 수 있는 대화 형식으로 변환합니다:

```python
def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},  # 프롬프트
            {"type" : "image", "image" : sample["image"]}  # 이미지
          ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["output"]}  # 정답 (GPT가 생성한 설명)
          ]
        },
    ]
    return { "messages" : conversation }
```

**역할**:
- 이미지와 텍스트를 하나의 대화 형식으로 결합
- 모델이 이미지를 보고 설명을 생성하도록 학습

#### 2) 학습 설정 (Cell 17)

`SFTTrainer`를 사용하여 실제 학습을 진행합니다:

```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(...),  # Vision 모델 전용 데이터 콜레이터
    train_dataset = converted_train_dataset,
    eval_dataset = converted_test_dataset[:64],
    args = SFTConfig(
        per_device_train_batch_size = 2,      # GPU당 배치 크기
        gradient_accumulation_steps = 4,       # 실제 배치 크기 = 2 × 4 = 8
        max_steps = 3000,                      # 총 학습 스텝
        learning_rate = 5e-5,                 # 학습률
        optim = "adamw_8bit",                 # 8bit 최적화 (메모리 절약)
        output_dir = "outputs",               # 모델 저장 경로
        report_to = "wandb",                  # WandB 로깅
        # Vision 모델 필수 설정
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
    ),
)
```

**주요 하이퍼파라미터**:
- `max_steps = 3000`: 학습 스텝 수 (조정 가능)
- `learning_rate = 5e-5`: 학습률 (일반적으로 1e-5 ~ 5e-5)
- `per_device_train_batch_size = 2`: GPU당 배치 크기
- `gradient_accumulation_steps = 4`: 실제 배치 크기 = batch_size × accumulation_steps

**배치 크기 설정 차이점**:
- `batch_size=2, accumulation=4` (실제 배치=8): GPU 메모리를 더 많이 사용하지만 학습 속도가 빠름
- `batch_size=1, accumulation=8` (실제 배치=8): GPU 메모리를 적게 사용하지만 학습 속도가 느림
- **GPU 메모리가 부족하면**: `batch_size=1, accumulation=8` 사용 권장
- **GPU 메모리가 충분하면**: `batch_size=2, accumulation=4` 사용 권장

**중요 설정**:
- `remove_unused_columns = False`: Vision 모델 필수
- `dataset_text_field = ""`: Vision 모델 필수
- `dataset_kwargs = {"skip_prepare_dataset": True}`: Vision 모델 필수

### 1-3. 노트북 실행

1. Jupyter Lab에서 `notebooks/dataset_study.ipynb` 열기
2. 셀을 순서대로 실행
3. 학습 완료까지 대기 (약 수 시간 소요)

**학습 진행 확인**:
- WandB 대시보드에서 실시간으로 학습 진행 상황 확인
- 터미널에서 로그 확인

### 1-4. 학습 결과 확인

- 학습된 모델은 `outputs/` 폴더에 저장됩니다
- WandB 대시보드에서 학습 진행 상황 확인 가능

### 1-5. 모델 업로드 (선택)

학습 완료 후 HuggingFace Hub에 업로드:

```python
# HuggingFace 로그인 (토큰 직접 입력)
!huggingface-cli login --token your_huggingface_token_here

# 본인의 HuggingFace 사용자명과 모델명으로 변경하세요
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

---

## Step 2: 모델 병합

### 2-1. 모델 경로 설정

`notebooks/model_merge.ipynb`의 Cell 2를 수정:

```python
# 노트북에서 직접 수정
HF_MODEL_NAME = "your-username/your-model-name"

# 또는 로컬 경로 사용 (Step 1에서 outputs/ 폴더에 저장된 경우)
# HF_MODEL_NAME = "./outputs"
```

### 2-2. 노트북 실행

1. `notebooks/model_merge.ipynb` 열기
2. 셀을 순서대로 실행
3. 병합 완료까지 대기

### 2-3. 결과 확인

- 병합된 모델은 `./model_16bit/` 폴더에 저장됩니다
- 이 모델을 vLLM 배포에 사용합니다

---

## Step 3: vLLM 배포 (중요)

### 3-1. vLLM 설치 및 배포

Pod 터미널에서 실행:

```bash
# 1. vLLM 설치
pip install -q vllm

# 2. 모델 배포 (한 줄로 실행)
# 기본 포트 8000 사용 (--port 옵션 생략 가능)
python3 -m vllm.entrypoints.openai.api_server --model ./model_16bit --dtype bfloat16 --tokenizer ./model_16bit
```

**배포 성공 확인**:
- `INFO: Uvicorn running on http://0.0.0.0:8000` 메시지가 보이면 성공
- 새 터미널에서 테스트: `curl http://localhost:8000/v1/models`

### 3-2. 자주 발생하는 오류 해결

#### GPU 메모리 부족 (가장 흔함)

**증상**: `CUDA out of memory`

**원인**: GPU VRAM이 48GB 미만

**해결**:
```bash
# 방법 1: 메모리 최적화 옵션 추가
python3 -m vllm.entrypoints.openai.api_server --model ./model_16bit --dtype bfloat16 --tokenizer ./model_16bit --max-model-len 2048 --gpu-memory-utilization 0.9

# 방법 2: float16 사용 (더 적은 메모리)
python3 -m vllm.entrypoints.openai.api_server --model ./model_16bit --dtype float16 --tokenizer ./model_16bit

# 방법 3: GPU 확인 후 48GB 미만이면 Pod 재생성 필요
nvidia-smi
```

#### 기타 오류 (일반적으로 발생할 수 있는 케이스)

**모델 경로 오류**:
```bash
# 증상: FileNotFoundError: './model_16bit'
# 해결: Step 2 완료 확인 및 절대 경로 사용
ls -la ./model_16bit/
python3 -m vllm.entrypoints.openai.api_server --model /workspace/model_16bit --dtype bfloat16 --tokenizer /workspace/model_16bit
```

**dtype 오류**:
```bash
# 증상: Unsupported dtype: bfloat16
# 해결: float16으로 변경
python3 -m vllm.entrypoints.openai.api_server --model ./model_16bit --dtype float16 --tokenizer ./model_16bit
```

**기타 오류 발생 시**:
- 오류 메시지를 자세히 확인하세요
- vLLM 공식 문서: https://docs.vllm.ai/

### 3-3. 배포 성공 확인

**성공 메시지**:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

**테스트** (새 터미널):
```bash
curl http://localhost:8000/v1/models
```

### 3-4. 포트 포워딩 (로컬 접속)

RunPod Pod → **"Connect"** → **"TCP Port Mapping"** → 포트 **8000** 매핑

---

## Step 4: 모델 테스트

### 4-1. 배치 테스트 (권장)

Step 3에서 vLLM 배포가 완료된 후, **새 터미널**에서 배치 테스트 스크립트를 실행합니다:

```bash
# 새 터미널에서 실행
python vllm_batch_test.py
```

**테스트 스크립트 기능**:
- 테스트 이미지들을 배치로 처리
- 정확도 및 응답 속도 측정
- 결과를 엑셀 파일로 저장 (`vllm_test_results.xlsx`)

**필요한 설정**:
- `vllm_batch_test.py`의 `base_url` 확인 (기본: `http://localhost:8000/v1`)
- 테스트 이미지 폴더 구조 확인 (`./test/VS_건선_정면/`, `./test/VS_아토피_정면/` 등)

### 4-2. Langchain 테스트 (선택)

Langchain을 사용한 테스트도 가능합니다:

```bash
# Langchain 설치
pip install -U langchain-openai

# 테스트 스크립트 실행
python vllm_langchain_test.py
```

**Langchain 테스트 스크립트** (`vllm_langchain_test.py`):
- Langchain의 `ChatOpenAI` 사용
- 이미지와 텍스트를 동시에 입력 (ChatCompletion 형태)
- 단일 이미지 테스트에 적합

**주의사항**:
- Image와 Text를 동시에 넣는 것은 **ChatCompletion** 형태여야 함
- TextCompletion으로는 실행되지 않음

### 4-3. 혼동행렬 평가 (선택)

상세한 성능 분석이 필요하면 `notebooks/confusion_matrix.ipynb`를 사용합니다:

1. `notebooks/confusion_matrix.ipynb`의 Cell 8 수정:
```python
API_BASE_URL = "http://localhost:8000/v1"  # vLLM 배포 포트에 맞춰 수정
API_KEY = "empty"
```

2. Cell 4 수정:
```python
HF_DATASET_NAME = "your-username/your-dataset-name"
```

3. 노트북 실행:
   - 혼동행렬 시각화
   - Accuracy, Precision, Recall, F1 Score
   - 클래스별 상세 리포트

---

## 문제 해결 팁

1. **오류 메시지 확인**: 첫 줄부터 읽기
   - `CUDA out of memory` → GPU VRAM 부족
   - `No such file` → 경로 문제
   - `Unsupported dtype` → dtype 변경

2. **단계별 확인**
   - Step 1: `outputs/` 폴더 확인
   - Step 2: `model_16bit/` 폴더 확인
   - Step 3: `curl http://localhost:8000/v1/models` 테스트

### GPU 선택 가이드

| GPU 모델 | VRAM | 학습 가능 | vLLM 배포 가능 |
|---------|------|----------|---------------|
| RTX A6000 | 48GB | 가능 | 가능 |
| A100 40GB | 40GB | 가능 | 제한적 |
| A100 80GB | 80GB | 가능 | 가능 |
| RTX A5000 | 24GB | 가능 | 불가 |
| RTX 4090 | 24GB | 가능 | 불가 |
※a100 모델은 오버 스펙이니 rtx a6000을 추천합니다.
---

## 파일 구조

```
finetuning/
├── README.md                    # 이 파일
└── notebooks/                    # 수정된 노트북 (개인정보 제거)
    ├── dataset_study.ipynb      # Step 1: 학습
    ├── model_merge.ipynb         # Step 2: 병합
    └── confusion_matrix.ipynb   # Step 4: 평가
```

---

## 주의사항

1. **Pod 비용**
   - 작업 완료 후 **반드시 Pod를 종료**하세요
   - Pod가 실행 중이면 계속 비용이 발생합니다

2. **모델 저장**
   - 중요한 모델은 HuggingFace Hub에 업로드하거나
   - Volume Disk에 저장하세요
   - Pod 종료 시 Container Disk의 데이터는 삭제될 수 있습니다

---

## 참고 자료

- [Unsloth 공식 문서](https://github.com/unslothai/unsloth)
- [vLLM 공식 문서](https://docs.vllm.ai/)
- [RunPod 문서](https://docs.runpod.io/)

---

## 체크리스트

파인튜닝 시작 전:

- [ ] RunPod Pod 생성 (GPU VRAM 48GB 이상)
- [ ] HuggingFace 토큰 준비
- [ ] WandB API 키 준비
- [ ] 데이터셋 이름 확인
- [ ] Container Disk 50GB 이상 설정

학습 완료 후:

- [ ] 모델이 `outputs/` 폴더에 저장되었는지 확인
- [ ] (선택) HuggingFace Hub에 업로드

병합 완료 후:

- [ ] `model_16bit/` 폴더 생성 확인
- [ ] 모델 파일들이 모두 있는지 확인

vLLM 배포 전:

- [ ] GPU VRAM 48GB 이상 확인
- [ ] 모델 경로 확인
- [ ] 디스크 여유 공간 확인

배포 성공 후:

- [ ] API 서버가 정상 작동하는지 테스트
- [ ] 평가 노트북 실행 준비

---

**문제가 발생하면 오류 메시지를 자세히 확인하고, 위의 해결 방법을 참고하세요!**

