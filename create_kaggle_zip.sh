#!/bin/bash

# Kaggle Dataset ZIP 파일 생성 스크립트

echo "📦 Creating Kaggle Dataset ZIP..."

# ZIP 파일명
ZIP_FILE="mydata.zip"

# 기존 ZIP 파일 삭제
if [ -f "$ZIP_FILE" ]; then
    echo "✓ Removing existing $ZIP_FILE..."
    rm "$ZIP_FILE"
fi

# ZIP 파일 생성 (폴더 구조 유지)
echo "✓ Creating ZIP file with folder structure..."
zip -r "$ZIP_FILE" \
    src/ \
    scripts/optimize_return_model.py \
    scripts/optimize_risk_model.py \
    scripts/optimize_position_strategy.py \
    scripts/optimize_ensemble.py \
    conf/params.yaml \
    -x "*.pyc" \
    -x "*__pycache__/*" \
    -x "*.git/*" \
    -x "*.DS_Store"

# 결과 확인
if [ -f "$ZIP_FILE" ]; then
    FILE_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
    echo ""
    echo "✅ ZIP file created successfully!"
    echo "📁 File: $ZIP_FILE"
    echo "📊 Size: $FILE_SIZE"
    echo ""
    echo "📦 포함된 파일:"
    echo "  - src/*.py (모든 Python 모듈)"
    echo "  - optimize_*.py (학습 스크립트 3개, 루트에 위치)"
    echo "  - params.yaml (설정 파일, 루트에 위치)"
    echo ""
    echo "�📌 다음 단계:"
    echo "1. https://www.kaggle.com/datasets 접속"
    echo "2. 'New Dataset' 클릭"
    echo "3. $ZIP_FILE 업로드"
    echo "4. Title: 'mydata' 입력"
    echo "5. 'Create' 클릭"
    echo "6. kaggle_inference_server.ipynb를 Kaggle Notebook으로 제출"
    echo ""
else
    echo "❌ Failed to create ZIP file"
    exit 1
fi
