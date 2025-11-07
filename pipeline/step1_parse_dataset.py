# -*- coding: utf-8 -*-
"""
Step 1: 이미지 파싱 및 DatasetDict 생성 (통합 버전)
- 이미지 폴더를 순회하며 데이터셋 생성
- label 필드에서 _정면, _측면 자동 제거
"""

import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, Image, DatasetDict

def parse_data(root_dir):
    """이미지 폴더를 순회하며 데이터셋 생성"""
    data = []
    root_path = Path(root_dir)
    
    for label_folder in root_path.iterdir():
        if not label_folder.is_dir():
            continue
        
        # 원본 label 추출 및 정리
        label = label_folder.name
        # TS_, VS_ 제거
        label = label.replace("TS_", "").replace("VS_", "")
        # _정면, _측면 제거 (통합된 기능)
        label = label.replace("_정면", "").replace("_측면", "")
        
        for img_file in label_folder.rglob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                data.append({
                    "image": {"path": str(img_file)},
                    "system_prompt": "You are a helpful vision assistant.",
                    "label": label,
                    "output": ""
                })
    
    print(f"  - {len(data)}개 이미지")
    
    df = pd.DataFrame(data)
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("image", Image())
    return ds


def main():
    print("=" * 60)
    print("Step 1: 데이터셋 파싱 (통합 버전)")
    print("=" * 60)
    
    # 경로 설정
    train_root_dir = "./extracted_data/Training/images"
    test_root_dir = "./extracted_data/Validation/images"
    SAVE_PATH = "./skin_dataset"
    
    # 1. 데이터 파싱
    print("\n데이터셋 파싱 중...")
    print("Training 데이터:")
    trainset = parse_data(train_root_dir)
    
    print("Validation 데이터:")
    testset = parse_data(test_root_dir)
    
    dataset = DatasetDict({"train": trainset, "test": testset})
    print(f"\n데이터셋 생성 완료!")
    print(dataset)
    
    # 2. 라벨 확인
    print("\n라벨 전처리 결과:")
    train_labels = sorted(list(set(dataset["train"]["label"])))
    print(f"Train labels: {train_labels}")
    
    # 3. 저장
    print(f"\n데이터셋 저장 중... ({SAVE_PATH})")
    dataset.save_to_disk(SAVE_PATH)
    print("저장 완료!")
    
    print("\n" + "=" * 60)
    print(f"저장 위치: {SAVE_PATH}")
    print("=" * 60)
    print("\n다음 단계:")
    print("   python pipeline/step2_add_descriptions.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

