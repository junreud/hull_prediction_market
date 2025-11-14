#!/bin/bash

# Kaggle Dataset 자동 업로드 스크립트

echo "📦 Creating Kaggle Dataset..."

# 1. 필요한 파일들을 임시 폴더에 복사
TEMP_DIR="kaggle_dataset_temp"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

echo "✓ Copying files..."
cp -r src/ $TEMP_DIR/
cp -r scripts/ $TEMP_DIR/
cp -r conf/ $TEMP_DIR/
cp -r artifacts/ $TEMP_DIR/

# 2. dataset-metadata.json 생성
cat > $TEMP_DIR/dataset-metadata.json << 'EOF'
{
  "title": "models",
  "id": "junreud/models",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo ""
echo "⚠️  dataset-metadata.json 파일을 수정하세요:"
echo "   'YOUR_USERNAME'을 본인의 Kaggle username으로 변경"
echo ""
echo "수정 후 다음 명령어 실행:"
echo "   cd $TEMP_DIR"
echo "   kaggle datasets create -p ."
echo ""
echo "또는 업데이트:"
echo "   kaggle datasets version -p . -m 'Updated models'"
