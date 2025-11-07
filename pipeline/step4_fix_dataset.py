# -*- coding: utf-8 -*-
"""
Step 4: 데이터셋 후처리 적용

config_postprocess.py의 설정을 기반으로:
- 영어 의학 용어 → 한글 번역
- output의 <label> 태그 통일
"""

import re
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
from config_postprocess import MEDICAL_TERMS, LABEL_MAPPING


def translate_text(text):
    """텍스트 내 영어 의학 용어를 한글로 번역"""
    if not text or not isinstance(text, str):
        return text
    
    translated = text
    for eng, kor in MEDICAL_TERMS.items():
        # 단어 경계를 고려한 패턴
        pattern = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
        translated = pattern.sub(kor, translated)
    
    return translated


def fix_output_label(output):
    """output 필드의 <label> 태그 내용 통일"""
    if not output:
        return output
    
    # <label>...</label> 찾기
    match = re.search(r'<label>(.*?)</label>', output)
    if match:
        old_label = match.group(1)
        # 매핑된 라벨로 변경
        new_label = LABEL_MAPPING.get(old_label, old_label)
        # output에서 교체
        output = output.replace(f'<label>{old_label}</label>', f'<label>{new_label}</label>')
    
    return output


def process_dataset(dataset_path, save_path):
    """데이터셋 후처리 적용"""
    
    print("=" * 80)
    print("Step 4: 데이터셋 후처리 적용")
    print("=" * 80)
    
    # 데이터셋 로드
    print(f"\n데이터셋 로드 중: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print("✓ 로드 완료")
    print(dataset)
    
    print("\n설정 정보:")
    print(f"  - 영어 용어 번역: {len(MEDICAL_TERMS)}개")
    print(f"  - 라벨 매핑: {len(LABEL_MAPPING)}개")
    
    # Train 데이터 처리
    print("\n" + "=" * 80)
    print("Train 데이터 후처리 중...")
    print("=" * 80)
    
    train_outputs = []
    train_changed = 0
    
    for example in tqdm(dataset["train"], desc="Train"):
        output = example['output']
        
        if not output:
            train_outputs.append(output)
            continue
        
        # 후처리 적용
        original = output
        output = translate_text(output)
        output = fix_output_label(output)
        
        if output != original:
            train_changed += 1
        
        train_outputs.append(output)
    
    print(f"✓ Train 완료: {train_changed}/{len(train_outputs)}개 수정됨")
    
    # Test 데이터 처리
    print("\n" + "=" * 80)
    print("Test 데이터 후처리 중...")
    print("=" * 80)
    
    test_outputs = []
    test_changed = 0
    
    for example in tqdm(dataset["test"], desc="Test"):
        output = example['output']
        
        if not output:
            test_outputs.append(output)
            continue
        
        # 후처리 적용
        original = output
        output = translate_text(output)
        output = fix_output_label(output)
        
        if output != original:
            test_changed += 1
        
        test_outputs.append(output)
    
    print(f"✓ Test 완료: {test_changed}/{len(test_outputs)}개 수정됨")
    
    # 데이터셋 업데이트
    print("\n" + "=" * 80)
    print("데이터셋 업데이트 중...")
    print("=" * 80)
    
    dataset["train"] = dataset["train"].remove_columns("output")
    dataset["train"] = dataset["train"].add_column("output", train_outputs)
    
    dataset["test"] = dataset["test"].remove_columns("output")
    dataset["test"] = dataset["test"].add_column("output", test_outputs)
    
    # 저장
    print(f"\n저장 중: {save_path}")
    dataset.save_to_disk(save_path)
    print("✓ 저장 완료!")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("처리 결과 요약")
    print("=" * 80)
    print(f"총 처리: {len(train_outputs) + len(test_outputs)}개")
    print(f"수정됨: {train_changed + test_changed}개")
    print(f"변경 없음: {len(train_outputs) + len(test_outputs) - train_changed - test_changed}개")
    print(f"\n저장 위치: {save_path}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("다음 단계:")
    print("  1. (선택) python pipeline/step3_check_issues.py 재실행으로 검증")
    print("  2. python pipeline/step5_upload_to_hub.py 실행으로 HuggingFace Hub 업로드")
    print("=" * 80)


def main():
    DATASET_PATH = "./skin_dataset"
    SAVE_PATH = "./skin_dataset_fixed"
    
    process_dataset(DATASET_PATH, SAVE_PATH)


if __name__ == "__main__":
    main()

