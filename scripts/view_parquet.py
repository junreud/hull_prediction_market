#!/usr/bin/env python3
"""
Parquet 파일 뷰어

Usage:
    python view_parquet.py <파일경로>
    python view_parquet.py data/raw/train.csv
"""

import sys
from pathlib import Path
import pandas as pd


def view_parquet(file_path: str):
    """Parquet 파일 내용 출력"""
    
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    print("="*80)
    print(f"📊 파일: {path.name}")
    print("="*80)
    
    try:
        # Parquet 또는 CSV 읽기
        if path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            print(f"⚠️  지원하지 않는 파일 형식: {path.suffix}")
            print("   .parquet 또는 .csv 파일만 지원합니다.")
            return
        
        # 기본 정보
        print(f"\n📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 컬럼 정보
        print(f"\n📋 Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) * 100
            print(f"  {i:2d}. {col:30s} | {str(dtype):10s} | Nulls: {null_count:6d} ({null_pct:5.1f}%)")
        
        # 처음 몇 행
        print(f"\n📄 First 10 rows:")
        print(df.head(10).to_string())
        
        # 기술 통계
        if len(df.select_dtypes(include=['number']).columns) > 0:
            print(f"\n📈 Numeric columns statistics:")
            print(df.describe().to_string())
        
        # 마지막 몇 행
        print(f"\n📄 Last 5 rows:")
        print(df.tail(5).to_string())
        
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_parquet.py <파일경로>")
        print("\nExamples:")
        print("  python view_parquet.py data/raw/train.csv")
        print("  python view_parquet.py submissions/submission.parquet")
        print("  python view_parquet.py artifacts/oof_predictions.parquet")
        sys.exit(1)
    
    file_path = sys.argv[1]
    view_parquet(file_path)
