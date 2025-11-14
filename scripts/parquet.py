"""
Parquet 파일 분석 스크립트
"""

import pandas as pd
from pathlib import Path

def analyze_parquet(file_path: str):
    """Parquet 파일 내용 분석"""
    
    print(f"📊 파일 분석: {file_path}")
    print("="*80)
    
    # 파일 읽기
    df = pd.read_parquet(file_path)
    
    # 기본 정보
    print(f"\n✓ Shape: {df.shape} (rows × columns)")
    print(f"\n✓ Columns: {list(df.columns)}")
    
    print(f"\n✓ Data types:")
    print(df.dtypes)
    
    # 처음 20행 출력
    print(f"\n✓ 처음 20행:")
    print(df.head(20))
    
    # 기본 통계
    print(f"\n✓ 기본 통계:")
    print(df.describe())
    
    # 결측치 확인
    print(f"\n✓ 결측치:")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(null_counts[null_counts > 0])
    else:
        print("결측치 없음")
    
    # 고유값 개수
    print(f"\n✓ 각 컬럼별 고유값 개수:")
    print(df.nunique())
    
    # allocation 고유값 확인
    if 'allocation' in df.columns:
        print(f"\n✓ allocation 고유값:")
        unique_vals = sorted(df['allocation'].unique())
        print(f"  값: {unique_vals}")
        print(f"\n✓ allocation 값 분포:")
        print(df['allocation'].value_counts().sort_index())
    
    return df


if __name__ == "__main__":
    # submission.parquet 파일 찾기
    parquet_files = list(Path('.').rglob('*.parquet'))
    
    if parquet_files:
        print("📁 발견된 parquet 파일들:")
        for f in parquet_files:
            print(f"  - {f}")
        
        print("\n")
        
        # 각 파일 분석
        for file_path in parquet_files:
            df = analyze_parquet(file_path)
            print("\n" + "="*80 + "\n")
    else:
        print("❌ parquet 파일을 찾을 수 없습니다.")
        print("\n특정 파일을 분석하려면:")
        print("  python parquet.py path/to/file.parquet")
