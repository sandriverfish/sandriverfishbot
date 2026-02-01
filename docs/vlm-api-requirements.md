# VLM Server API Requirements

**Version:** 1.0
**Date:** 2025-02-01
**Status:** Specification for VLM Server Team

This document specifies the API changes/additions required for the VLM Server at `192.168.1.196:8080` to support the BestBox mold troubleshooting system integration.

---

## Table of Contents

1. [Overview](#overview)
2. [Required: Multipart File Upload Endpoint](#required-multipart-file-upload-endpoint)
3. [Required: Mold-Specific Analysis Prompt Template](#required-mold-specific-analysis-prompt-template)
4. [Optional: Batch Processing Enhancement](#optional-batch-processing-enhancement)
5. [Optional: Defect Similarity API](#optional-defect-similarity-api)
6. [Webhook Callback Specification](#webhook-callback-specification)
7. [Error Handling](#error-handling)

---

## Overview

The BestBox mold troubleshooting system needs to integrate with the Qwen3-VL 8B service for:

1. **Batch Knowledge Base Enrichment** - Process documents/images during indexing
2. **Real-time Chat Analysis** - Analyze user-uploaded images during conversations
3. **Defect Pattern Comparison** - Correlate new issues with historical cases

### Current Limitation

The existing API only supports `file_url` (fetch from URL). Since the BestBox server runs on a private network that the VLM server cannot access, we need **direct file upload via multipart form data**.

---

## Required: Multipart File Upload Endpoint

### Endpoint

```
POST /api/v1/jobs/upload
Content-Type: multipart/form-data
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary | Yes | The file to analyze (images, PDF, documents) |
| `webhook_url` | string | No | Callback URL for completion notification |
| `prompt_template` | string | No | Built-in template ID or custom prompt |
| `options` | JSON string | No | Processing options (see below) |

### Options Object

```json
{
  "extract_images": true,
  "analysis_depth": "detailed",
  "output_language": "zh",
  "max_tokens": 2048,
  "include_ocr": true
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extract_images` | boolean | true | Extract and analyze embedded images |
| `analysis_depth` | string | "standard" | "quick", "standard", or "detailed" |
| `output_language` | string | "zh" | Output language ("zh", "en") |
| `max_tokens` | integer | 1024 | Max output tokens |
| `include_ocr` | boolean | true | Extract text from images |

### Example Request

```bash
curl -X POST http://192.168.1.196:8080/api/v1/jobs/upload \
  -F "file=@document.pdf" \
  -F "webhook_url=http://192.168.1.100:8000/api/v1/webhooks/vlm-results" \
  -F "prompt_template=mold_defect_analysis" \
  -F 'options={"extract_images": true, "analysis_depth": "detailed", "output_language": "zh"}'
```

### Response

```json
{
  "job_id": "vlm-20250201-abc123",
  "status": "pending",
  "estimated_duration": "45s",
  "check_status_url": "http://192.168.1.196:8080/api/v1/jobs/vlm-20250201-abc123",
  "submitted_at": "2025-02-01T10:30:00Z"
}
```

### Job Status Endpoint

```
GET /api/v1/jobs/{job_id}
```

**Response (pending):**
```json
{
  "job_id": "vlm-20250201-abc123",
  "status": "processing",
  "progress": 0.6,
  "estimated_remaining": "15s"
}
```

**Response (completed):**
```json
{
  "job_id": "vlm-20250201-abc123",
  "status": "completed",
  "result": {
    "document_summary": "该文档记录了注塑模具的试模问题...",
    "key_insights": [
      "产品存在披锋问题，主要在分型线位置",
      "表面有火花纹残留，需要重新抛光"
    ],
    "analysis": {
      "sentiment": "neutral",
      "topics": ["披锋", "表面质量", "模具维护"],
      "entities": ["ED736A0501", "T2试模"],
      "complexity_score": 0.7
    },
    "extracted_images": [
      {
        "image_id": "img001",
        "page": 1,
        "description": "产品侧面照片，显示分型线位置有明显披锋",
        "insights": "披锋高度约0.3mm，需要调整锁模力或检查分型面配合",
        "defect_type": "披锋",
        "bounding_box": {"x": 120, "y": 80, "width": 200, "height": 150}
      }
    ],
    "text_content": "提取的文档文字内容...",
    "tags": ["注塑", "披锋", "T2试模", "表面缺陷"],
    "metadata": {
      "confidence_score": 0.92,
      "processing_model": "Qwen3-VL-8B",
      "tokens_used": 1847,
      "processing_time_ms": 12500
    }
  },
  "completed_at": "2025-02-01T10:30:45Z"
}
```

---

## Required: Mold-Specific Analysis Prompt Template

### Template ID: `mold_defect_analysis`

Register this as a built-in prompt template that can be referenced by ID.

### Template Content

```
Analyze this manufacturing/mold-related document or image. You are an expert in injection molding defect diagnosis.

Extract the following information in structured JSON format:

1. **Defect Types** (缺陷类型): Identify any visible defects:
   - 披锋/flash (material overflow at parting line)
   - 拉白/whitening (stress whitening)
   - 火花纹/spark marks (EDM marks not polished)
   - 脏污/contamination (surface contamination)
   - 划痕/scratches
   - 变形/deformation
   - 缩水/sink marks
   - 熔接痕/weld lines

2. **Equipment Parts** (设备部件): Identify mold components visible:
   - 动模/moving half
   - 定模/fixed half
   - 型腔/cavity
   - 型芯/core
   - 滑块/slider
   - 顶针/ejector pin
   - 浇口/gate
   - 流道/runner

3. **Text Content** (文字内容): Extract any visible text:
   - Part numbers (零件号)
   - Trial versions (T0/T1/T2)
   - Annotations and markings
   - Handwritten notes

4. **Visual Annotations** (视觉标注): Note any:
   - Arrows indicating problem areas
   - Circles or highlights
   - Comparison before/after markings

5. **Root Cause Indicators** (根本原因线索):
   - Visible clues about defect causes
   - Suggested corrective actions

6. **Severity Assessment** (严重程度):
   - high: Production blocking, customer impact
   - medium: Requires action before mass production
   - low: Minor issue, can be addressed in maintenance

Output format:
{
  "defect_type": "string (primary defect category in Chinese)",
  "defect_details": "string (detailed description)",
  "equipment_part": "string (identified equipment/mold part)",
  "text_in_image": "string (all extracted text)",
  "visual_annotations": "string (description of markings)",
  "severity": "string (high/medium/low)",
  "root_cause_hints": ["array of strings"],
  "suggested_actions": ["array of corrective action strings"],
  "confidence": 0.0-1.0
}
```

---

## Optional: Batch Processing Enhancement

For processing multiple images from a single troubleshooting case simultaneously.

### Endpoint

```
POST /api/v1/jobs/batch
Content-Type: multipart/form-data
```

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files[]` | binary[] | Yes | Multiple files to analyze |
| `webhook_url` | string | No | Callback URL for completion |
| `options` | JSON string | No | Processing options |

### Options for Batch

```json
{
  "cross_reference": true,
  "identify_similar_defects": true,
  "group_by_defect_type": true
}
```

### Example Request

```bash
curl -X POST http://192.168.1.196:8080/api/v1/jobs/batch \
  -F "files[]=@image1.jpg" \
  -F "files[]=@image2.jpg" \
  -F "files[]=@image3.jpg" \
  -F "webhook_url=http://192.168.1.100:8000/api/v1/webhooks/vlm-results" \
  -F 'options={"cross_reference": true, "identify_similar_defects": true}'
```

### Response

```json
{
  "batch_job_id": "vlm-batch-20250201-xyz789",
  "status": "pending",
  "file_count": 3,
  "estimated_duration": "90s",
  "check_status_url": "http://192.168.1.196:8080/api/v1/jobs/vlm-batch-20250201-xyz789"
}
```

### Batch Result Format

```json
{
  "batch_job_id": "vlm-batch-20250201-xyz789",
  "status": "completed",
  "results": [
    {
      "file_index": 0,
      "filename": "image1.jpg",
      "analysis": { /* standard analysis result */ }
    },
    {
      "file_index": 1,
      "filename": "image2.jpg",
      "analysis": { /* standard analysis result */ }
    }
  ],
  "cross_reference": {
    "common_defects": ["披锋"],
    "defect_groups": [
      {
        "defect_type": "披锋",
        "occurrences": [0, 2],
        "similarity_score": 0.85
      }
    ]
  }
}
```

---

## Optional: Defect Similarity API

For comparing a new image against reference images to find similar defect patterns.

### Endpoint

```
POST /api/v1/compare
Content-Type: multipart/form-data
```

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reference_image` | binary | Yes | The reference image |
| `comparison_images[]` | binary[] | Yes | Images to compare against reference |
| `comparison_type` | string | No | Type of comparison (default: "defect_similarity") |

### Comparison Types

- `defect_similarity` - Compare defect patterns
- `equipment_match` - Match equipment/mold parts
- `visual_similarity` - General visual similarity

### Example Request

```bash
curl -X POST http://192.168.1.196:8080/api/v1/compare \
  -F "reference_image=@new_defect.jpg" \
  -F "comparison_images[]=@historical1.jpg" \
  -F "comparison_images[]=@historical2.jpg" \
  -F "comparison_type=defect_similarity"
```

### Response

```json
{
  "job_id": "vlm-compare-20250201-def456",
  "status": "completed",
  "reference_analysis": {
    "defect_type": "披锋",
    "severity": "medium",
    "location": "分型线位置"
  },
  "similarities": [
    {
      "image_index": 0,
      "filename": "historical1.jpg",
      "similarity_score": 0.85,
      "matching_defects": ["披锋", "表面划痕"],
      "matching_regions": [
        {"ref_box": [120, 80, 200, 150], "comp_box": [110, 75, 210, 160]}
      ],
      "differences": ["位置略有不同", "严重程度不同"]
    },
    {
      "image_index": 1,
      "filename": "historical2.jpg",
      "similarity_score": 0.42,
      "matching_defects": [],
      "differences": ["缺陷类型不同"]
    }
  ],
  "recommendations": [
    "与历史案例 historical1.jpg 高度相似，建议参考该案例的解决方案",
    "检查分型面配合精度",
    "调整锁模力参数"
  ]
}
```

---

## Webhook Callback Specification

When a job completes, if `webhook_url` was provided, send a POST request.

### Request

```
POST {webhook_url}
Content-Type: application/json
X-VLM-Signature: sha256=<HMAC-SHA256 of body using shared secret>
X-VLM-Job-ID: vlm-20250201-abc123
X-VLM-Timestamp: 2025-02-01T10:30:45Z
```

### Body

```json
{
  "event": "job.completed",
  "job_id": "vlm-20250201-abc123",
  "status": "completed",
  "result": { /* full job result */ },
  "completed_at": "2025-02-01T10:30:45Z"
}
```

### Signature Verification

The signature is computed as:
```
HMAC-SHA256(request_body, shared_secret)
```

The shared secret is configured during service setup.

### Retry Policy

If webhook delivery fails:
1. Retry after 5 seconds
2. Retry after 30 seconds
3. Retry after 2 minutes
4. Mark as failed, result available via GET endpoint

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 202 | Job accepted, processing async |
| 400 | Bad request (invalid file, missing params) |
| 413 | File too large (max 100MB) |
| 415 | Unsupported file type |
| 429 | Rate limited |
| 500 | Server error |
| 503 | Service unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "Unsupported file type: .docx. Supported: jpg, jpeg, png, webp, pdf, xlsx",
    "details": {
      "received_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "supported_types": ["image/jpeg", "image/png", "image/webp", "application/pdf", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
    }
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_FILE_TYPE` | Unsupported file format |
| `FILE_TOO_LARGE` | File exceeds size limit |
| `INVALID_OPTIONS` | Malformed options JSON |
| `TEMPLATE_NOT_FOUND` | Unknown prompt template ID |
| `PROCESSING_FAILED` | VLM processing error |
| `TIMEOUT` | Processing exceeded time limit |
| `RATE_LIMITED` | Too many requests |

---

## Health Check

### Endpoint

```
GET /api/v1/health
```

### Response

```json
{
  "status": "healthy",
  "model": "Qwen3-VL-8B",
  "version": "1.0.0",
  "gpu_memory_used": "7.2GB",
  "gpu_memory_total": "24GB",
  "queue_depth": 3,
  "average_processing_time_ms": 12500
}
```

---

## Appendix: Supported File Types

| Type | Extensions | Max Size |
|------|------------|----------|
| Images | jpg, jpeg, png, webp | 20MB |
| PDF | pdf | 50MB |
| Excel | xlsx | 50MB |
| Documents | txt, md | 10MB |

---

## Implementation Priority

1. **Phase 1 (Required):**
   - Multipart file upload endpoint
   - Mold-specific prompt template

2. **Phase 2 (Optional):**
   - Batch processing endpoint
   - Defect similarity comparison

---

## Contact

For questions about this specification, contact the BestBox development team.
