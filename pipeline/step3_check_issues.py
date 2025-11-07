# -*- coding: utf-8 -*-
"""
Step 3: 데이터셋 문제점 자동 탐지

기능:
- 영어 단어 자동 탐지
- 다른 언어(한글/영어 외) 탐지
- 라벨 불일치 체크 (label 필드 vs <label> 태그)
- 통계 리포트 출력
"""

import re
from collections import Counter
from datasets import load_from_disk

def find_english_words(text):
    """영어 단어 찾기 (3자 이상)"""
    if not text or not isinstance(text, str):
        return []
    # 알파벳 3자 이상 연속
    pattern = re.compile(r'\b[A-Za-z]{3,}\b')
    return pattern.findall(text)


def extract_label_from_output(output):
    """output에서 <label> 태그 추출"""
    if not output:
        return None
    match = re.search(r'<label>(.*?)</label>', output)
    return match.group(1) if match else None


def check_dataset_issues(dataset_path):
    """데이터셋 문제점 탐지"""
    
    print("=" * 80)
    print("Step 3: 데이터셋 문제점 자동 탐지")
    print("=" * 80)
    
    # 데이터셋 로드
    print(f"\n데이터셋 로드 중: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print("✓ 로드 완료")
    print(dataset)
    
    # 통계 변수
    all_english_words = []
    label_mismatches = []
    output_labels = []
    
    # Train + Test 전체 확인
    for split in ["train", "test"]:
        print(f"\n{split.upper()} 데이터 분석 중...")
        
        for i, example in enumerate(dataset[split]):
            output = example.get('output', '')
            label = example.get('label', '')
            
            if not output:
                continue
            
            # 1. 영어 단어 탐지
            english_words = find_english_words(output)
            all_english_words.extend(english_words)
            
            # 2. output의 <label> 추출
            output_label = extract_label_from_output(output)
            if output_label:
                output_labels.append(output_label)
                
                # 3. label 필드와 output <label> 불일치 체크
                if output_label != label:
                    label_mismatches.append({
                        'split': split,
                        'index': i,
                        'field_label': label,
                        'output_label': output_label
                    })
    
    # ========== 리포트 출력 ==========
    
    print("\n" + "=" * 80)
    print("📊 분석 결과 리포트")
    print("=" * 80)
    
    # 1. 영어 단어 통계
    if all_english_words:
        word_counts = Counter(all_english_words)
        print("\n" + "=" * 80)
        print("🔤 발견된 영어 단어 (빈도순 Top 30)")
        print("=" * 80)
        for word, count in word_counts.most_common(30):
            print(f"  - {word:30s} ({count}회)")
        
        print(f"\n총 {len(word_counts)}개의 고유 영어 단어 발견")
        print(f"총 {len(all_english_words)}회 출현")
    else:
        print("\n✅ 영어 단어 없음")
    
    # 2. output 라벨 통계
    if output_labels:
        label_counts = Counter(output_labels)
        print("\n" + "=" * 80)
        print("🏷️  output의 <label> 태그 분포")
        print("=" * 80)
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  - {label:20s} ({count}개)")
        
        print(f"\n총 {len(label_counts)}개의 고유 라벨")
    
    # 3. 라벨 불일치
    if label_mismatches:
        mismatch_summary = {}
        for item in label_mismatches:
            key = (item['field_label'], item['output_label'])
            mismatch_summary[key] = mismatch_summary.get(key, 0) + 1
        
        print("\n" + "=" * 80)
        print("⚠️  라벨 불일치 (label 필드 vs <label> 태그)")
        print("=" * 80)
        for (field_label, output_label), count in sorted(mismatch_summary.items(), key=lambda x: -x[1]):
            print(f"  - label='{field_label}' vs <label>{output_label}</label> ({count}건)")
        
        print(f"\n총 {len(label_mismatches)}건의 불일치")
    else:
        print("\n✅ 라벨 불일치 없음")
    
    # ========== 권장 사항 ==========
    
    print("\n" + "=" * 80)
    print("💡 권장 사항")
    print("=" * 80)
    
    if all_english_words:
        print("\n1️⃣  config_postprocess.py의 MEDICAL_TERMS에 다음 항목 추가 권장:")
        print("-" * 80)
        for word, count in word_counts.most_common(10):
            print(f'    "{word}": "한글_번역",  # {count}회')
    
    if label_mismatches:
        print("\n2️⃣  config_postprocess.py의 LABEL_MAPPING에 다음 항목 추가 권장:")
        print("-" * 80)
        unique_output_labels = set(item['output_label'] for item in label_mismatches)
        for output_label in sorted(unique_output_labels):
            print(f"    '{output_label}': '표준라벨',")
    
    print("\n" + "=" * 80)
    print("다음 단계:")
    print("  1. config_postprocess.py를 열어서 위 권장 사항 반영")
    print("  2. python pipeline/step4_fix_dataset.py 실행")
    print("=" * 80)


def main():
    DATASET_PATH = "./skin_dataset"
    check_dataset_issues(DATASET_PATH)


if __name__ == "__main__":
    main()

