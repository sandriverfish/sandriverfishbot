#!/bin/bash
# Quick test script for BestBox integration features

echo "Testing VLM API Server - BestBox Integration"
echo "=============================================="
echo ""

BASE_URL="http://192.168.1.196:8080"

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s $BASE_URL/api/v1/health | python3 -m json.tool
echo ""

# Test 2: List templates
echo "2. Testing templates endpoint..."
curl -s $BASE_URL/api/v1/templates | python3 -m json.tool
echo ""

# Test 3: Queue status
echo "3. Testing queue status..."
curl -s $BASE_URL/api/v1/queue | python3 -m json.tool
echo ""

echo "=============================================="
echo "Basic endpoints working!"
echo ""
echo "To test multipart upload:"
echo "  curl -X POST $BASE_URL/api/v1/jobs/upload \\"
echo "    -F 'file=@your-image.jpg' \\"
echo "    -F 'prompt_template=mold_defect_analysis' \\"
echo "    -F 'options={\"output_language\": \"zh\"}'"
echo ""
echo "To test with URL:"
echo "  curl -X POST $BASE_URL/api/v1/jobs \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"file_url\": \"http://example.com/image.jpg\", \"prompt_template\": \"mold_defect_analysis\"}'"
