# -*- coding: utf-8 -*-
"""
Step 5: HuggingFace Hub에 데이터셋 업로드

최종 전처리가 완료된 데이터셋을 HuggingFace Hub에 업로드합니다.
"""

import os
from datasets import load_from_disk
from dotenv import load_dotenv


def upload_to_hub(dataset_path, repo_name=None):
    """데이터셋을 HuggingFace Hub에 업로드"""
    
    print("=" * 80)
    print("Step 5: HuggingFace Hub 업로드")
    print("=" * 80)
    
    # 환경변수 로드
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("\n⚠️  HF_TOKEN이 설정되지 않았습니다.")
        hf_token = input("HuggingFace Token을 입력하세요: ").strip()
        if not hf_token:
            print("❌ 토큰이 필요합니다. 종료합니다.")
            return
    
    # 레포지토리 이름
    if not repo_name:
        repo_name = os.getenv("HF_REPO_NAME")
        if not repo_name:
            repo_name = input("HuggingFace 레포지토리 이름 (예: username/dataset-name): ").strip()
            if not repo_name:
                print("❌ 레포지토리 이름이 필요합니다. 종료합니다.")
                return
    
    # 데이터셋 로드
    print(f"\n데이터셋 로드 중: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print("✓ 로드 완료")
    print(dataset)
    
    # 데이터셋 정보 확인
    print("\n" + "=" * 80)
    print("업로드할 데이터셋 정보")
    print("=" * 80)
    print(f"Train: {len(dataset['train'])}개")
    print(f"Test: {len(dataset['test'])}개")
    
    # 라벨 분포 확인
    train_labels = dataset['train']['label']
    from collections import Counter
    label_dist = Counter(train_labels)
    print("\n클래스 분포 (Train):")
    for label, count in sorted(label_dist.items()):
        print(f"  - {label}: {count}개")
    
    # 확인
    print(f"\n레포지토리: {repo_name}")
    confirm = input("\n업로드하시겠습니까? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("업로드를 취소했습니다.")
        return
    
    # 업로드
    print("\n" + "=" * 80)
    print("업로드 중... (시간이 걸릴 수 있습니다)")
    print("=" * 80)
    
    try:
        dataset.push_to_hub(
            repo_name,
            token=hf_token,
            private=False  # 공개 데이터셋
        )
        print("\n✅ 업로드 완료!")
        print(f"\n🌐 데이터셋 URL: https://huggingface.co/datasets/{repo_name}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 업로드 실패: {str(e)}")
        print("\n확인 사항:")
        print("  - HF_TOKEN이 올바른지 확인")
        print("  - 레포지토리 이름 형식이 'username/dataset-name'인지 확인")
        print("  - 인터넷 연결 확인")


def main():
    DATASET_PATH = "./skin_dataset_fixed"
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 데이터셋을 찾을 수 없습니다: {DATASET_PATH}")
        print("먼저 step4_fix_dataset.py를 실행하세요.")
        return
    
    upload_to_hub(DATASET_PATH)


if __name__ == "__main__":
    main()

