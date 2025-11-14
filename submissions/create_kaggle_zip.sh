#!/bin/bash

# Create ZIP file for Kaggle Dataset Upload
# This script packages src/ and conf/ folders for Kaggle
# ./create_kaggle_zip.sh  <-- run this command in terminal

echo "========================================"
echo "Creating Kaggle Dataset ZIP"
echo "========================================"

# Define output file
OUTPUT_FILE="prediction_market_modules.zip"

# Remove old ZIP if exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing old ZIP file..."
    rm "$OUTPUT_FILE"
fi

# Create ZIP file
echo "Creating ZIP file..."
zip -r "$OUTPUT_FILE" src/ scripts/ conf/ data/raw/ \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*.pyo" \
    -x "*/.DS_Store" \
    -x "src/__pycache__/*" \
    -x "scripts/__pycache__/*" \
    -x "*.log"

# Check if successful
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✅ SUCCESS!"
    echo "========================================"
    echo "ZIP file created: $OUTPUT_FILE"
    echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "Contents:"
    unzip -l "$OUTPUT_FILE"
    echo ""
    echo "========================================"
    echo "Next steps:"
    echo "1. Go to https://www.kaggle.com/datasets"
    echo "2. Click 'New Dataset'"
    echo "3. Upload: $OUTPUT_FILE"
    echo "4. Name it: prediction-market-modules"
    echo "5. Click 'Create'"
    echo ""
    echo "Then in your Kaggle notebook:"
    echo "- Add Data → Your Datasets → prediction-market-modules"
    echo "- Run kaggle_submission_with_modules.ipynb"
    echo "========================================"
else
    echo ""
    echo "❌ FAILED to create ZIP file"
    echo "========================================"
    exit 1
fi
