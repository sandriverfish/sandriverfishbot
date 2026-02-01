# API Integration Guide for External Services

## Overview

This document provides API specifications for external servers and services to integrate with the local Qwen3-VL 8B vision-language model running on **192.168.1.196:8080**. Given the processing speed of **5-6 tokens/second**, all operations use **asynchronous processing with callbacks**.

**Use Case**: External services with their own LLM/embedding infrastructure can offload visual analysis tasks to this VLM, receiving structured JSON output for knowledge base construction.

**Base URL**: `http://192.168.1.196:8080`

---

## Architecture

**Processing Flow:**
1. External server submits file(s) via HTTP POST to `/api/v1/jobs`
2. API returns immediately with `job_id` and `status: pending`
3. VLM processes file asynchronously (extracts images, generates insights)
4. Upon completion, POST callback to external server's webhook URL
5. External server retrieves results via GET endpoint

**Token Speed Considerations:**
- Expected processing time: ~30-60 seconds per document page
- Large documents automatically chunked into pages
- Concurrent job limit: 3 (configurable)

---

## API Endpoints

### 1. Submit Job

**POST** `/api/v1/jobs`

Submit a document or image for asynchronous processing.

**Request Body:**
```json
{
  "file_url": "https://external-server.com/document.pdf",
  "webhook_url": "https://external-server.com/webhooks/vlm-results",
  "prompt_template": "optional-custom-prompt",
  "options": {
    "extract_images": true,
    "generate_summary": true,
    "analysis_depth": "detailed",
    "output_language": "en"
  }
}
```

**Response:**
```json
{
  "job_id": "vlm-20250201-001",
  "status": "pending",
  "estimated_duration": "45s",
  "check_status_url": "http://192.168.1.196:8080/api/v1/jobs/vlm-20250201-001",
  "submitted_at": "2025-02-01T10:30:00Z"
}
```

### 2. Check Job Status

**GET** `/api/v1/jobs/{job_id}`

Poll for job status (alternative to webhooks).

**Response (Pending):**
```json
{
  "job_id": "vlm-20250201-001",
  "status": "processing",
  "progress": 65,
  "tokens_generated": 450,
  "started_at": "2025-02-01T10:30:05Z"
}
```

**Response (Complete):**
```json
{
  "job_id": "vlm-20250201-001",
  "status": "completed",
  "completed_at": "2025-02-01T10:31:20Z",
  "result_url": "http://192.168.1.196:8080/api/v1/jobs/vlm-20250201-001/result"
}
```

### 3. Get Results

**GET** `/api/v1/jobs/{job_id}/result`

Retrieve the processed output.

**Response Structure:**
```json
{
  "job_id": "vlm-20250201-001",
  "input": {
    "filename": "annual_report_2024.pdf",
    "file_type": "application/pdf",
    "pages_processed": 12,
    "processing_duration": "75s"
  },
  "output": {
    "document_summary": "Comprehensive annual report covering...",
    "key_insights": [
      "Revenue increased by 23% year-over-year",
      "New product line launched in Q3",
      "Expansion into 3 new markets"
    ],
    "analysis": {
      "sentiment": "positive",
      "complexity_score": 7.2,
      "topics": ["finance", "growth", "strategy"],
      "entities": [
        {"name": "Company XYZ", "type": "organization", "mentions": 15},
        {"name": "CEO John Smith", "type": "person", "mentions": 8}
      ]
    },
    "extracted_images": [
      {
        "image_id": "img_001",
        "page": 3,
        "description": "Bar chart showing quarterly revenue trends",
        "insights": "Q4 shows significant growth spike...",
        "reference_url": "http://192.168.1.196:8080/api/v1/images/vlm-20250201-001/img_001"
      }
    ],
    "tags": ["financial", "annual-report", "2024", "growth", "charts"],
    "metadata": {
      "confidence_score": 0.89,
      "processing_model": "qwen3-vl-8b-q8",
      "tokens_used": 2847
    }
  }
}
```

### 4. Health Check

**GET** `/api/v1/health`

Check server status and metrics.

**Response:**
```json
{
  "status": "healthy",
  "model": "qwen3-vl-8b-q8",
  "gpu_available": true,
  "queue_length": 2,
  "active_jobs": 1,
  "avg_tokens_per_sec": 5.2
}
```

---

## Webhook Callback Format

When processing completes, the VLM server POSTs to your webhook URL:

```json
{
  "event": "job.completed",
  "job_id": "vlm-20250201-001",
  "timestamp": "2025-02-01T10:31:20Z",
  "result_summary": {
    "status": "success",
    "pages_processed": 12,
    "images_extracted": 4,
    "result_url": "http://192.168.1.196:8080/api/v1/jobs/vlm-20250201-001/result"
  }
}
```

**Webhook Security:**
- Include signature header: `X-VLM-Signature: sha256=<hmac>`
- Verify signature using shared secret
- Retry failed deliveries up to 3 times with exponential backoff

---

## Custom Prompt Templates

External servers can provide custom prompts for specific analysis needs:

**Example: Technical Documentation Analysis**
```json
{
  "prompt_template": "Analyze this technical document and extract:\n1. All API endpoints mentioned\n2. Code examples with language identification\n3. Configuration parameters\n4. Error codes and their meanings\n\nFormat output as structured JSON.",
  "options": {
    "extract_images": true,
    "focus_areas": ["code", "configuration", "api"]
  }
}
```

**Example: Image Analysis Only**
```json
{
  "prompt_template": "Describe each image in detail. Identify objects, text content, relationships between elements, and any anomalies or notable features.",
  "options": {
    "extract_images": true,
    "generate_summary": false
  }
}
```

---

## Rate Limits & Performance

**Current Constraints:**
- **Token speed**: 5-6 tokens/second
- **Max concurrent jobs**: 3
- **Average processing time**:
  - Single image: 15-30 seconds
  - 10-page document: 2-3 minutes
  - 50-page document: 10-15 minutes (batched)

**Queue Behavior:**
- Jobs exceeding limit are queued (FIFO)
- Queue status available at `GET /api/v1/queue`
- Priority jobs can be submitted with `"priority": "high"`

---

## Error Handling

**Async Error Response:**
```json
{
  "job_id": "vlm-20250201-002",
  "status": "failed",
  "error": {
    "code": "PROCESSING_TIMEOUT",
    "message": "Document processing exceeded 5 minute limit",
    "details": "Page 47 contains corrupted image data",
    "retryable": false
  },
  "failed_at": "2025-02-01T10:35:00Z"
}
```

**Common Error Codes:**
- `INVALID_FILE_FORMAT` - Unsupported file type
- `DOWNLOAD_FAILED` - Could not fetch file from URL
- `PROCESSING_TIMEOUT` - Exceeded time limit
- `RATE_LIMIT_EXCEEDED` - Too many concurrent requests
- `OUT_OF_MEMORY` - Document too large for available VRAM

---

## Integration Example (Python)

```python
import requests
import time

VLM_API_URL = "http://192.168.1.196:8080/api/v1"
WEBHOOK_URL = "https://myserver.com/webhooks/vlm"

def submit_document(file_url):
    """Submit document for async processing"""
    response = requests.post(f"{VLM_API_URL}/jobs", json={
        "file_url": file_url,
        "webhook_url": WEBHOOK_URL,
        "options": {
            "extract_images": True,
            "generate_summary": True,
            "analysis_depth": "detailed"
        }
    })
    
    job = response.json()
    print(f"Job submitted: {job['job_id']}")
    return job['job_id']

def poll_until_complete(job_id, timeout=600):
    """Poll for job completion (fallback to webhooks)"""
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(f"{VLM_API_URL}/jobs/{job_id}")
        data = response.json()
        
        if data['status'] == 'completed':
            return get_results(job_id)
        elif data['status'] == 'failed':
            raise Exception(f"Job failed: {data['error']}")
        
        print(f"Progress: {data.get('progress', 0)}%")
        time.sleep(5)
    
    raise TimeoutError("Job processing timeout")

def get_results(job_id):
    """Retrieve processed results"""
    response = requests.get(f"{VLM_API_URL}/jobs/{job_id}/result")
    result = response.json()
    
    # Store in knowledge base
    store_in_knowledge_base(result)
    return result

def store_in_knowledge_base(vlm_result):
    """Integrate with your existing knowledge base"""
    # Extract images for your own processing
    for img in vlm_result['output']['extracted_images']:
        # Download and store image
        img_data = requests.get(img['reference_url']).content
        save_to_storage(img['image_id'], img_data)
    
    # Store JSON metadata in your vector DB
    index_document(
        id=vlm_result['job_id'],
        content=vlm_result['output']['document_summary'],
        metadata=vlm_result['output']['metadata'],
        tags=vlm_result['output']['tags']
    )

# Usage
job_id = submit_document("https://myserver.com/docs/report.pdf")
result = poll_until_complete(job_id)
print(f"Processed {result['input']['pages_processed']} pages")
print(f"Extracted {len(result['output']['extracted_images'])} images")
```

---

## Network Configuration

**For External Server Access:**

The VLM server runs on **192.168.1.196:8080**. For external servers to reach it:

1. **Same Network**: Use local IP `http://192.168.1.196:8080`
2. **Different Networks**: Configure port forwarding or VPN
3. **Firewall**: Ensure port 8080 is open on Jetson

**Test Connectivity:**
```bash
curl http://192.168.1.196:8080/api/v1/health
```

**Security Recommendation:**
```bash
# Use SSH tunnel for secure remote access
ssh -L 8080:localhost:8080 user@192.168.1.196
# Then access via http://localhost:8080 on external server
```

---

## Best Practices

1. **Always use webhooks** rather than polling for production
2. **Implement idempotency** - VLM may retry failed callbacks
3. **Set reasonable timeouts** - Large documents take 5-15 minutes
4. **Cache results** - Store processed outputs to avoid re-processing
5. **Chunk large documents** - Submit 20-50 page segments for better reliability
6. **Monitor queue depth** - Scale down submissions when queue > 10
7. **Validate file URLs** - Ensure VLM server can reach your file server
8. **Handle partial failures** - Some pages may fail while others succeed

---

## Troubleshooting

**Slow Processing:**
- Check GPU utilization: `nvidia-smi` on Jetson
- Reduce concurrent jobs to 1-2
- Use smaller context windows for faster generation

**Connection Refused:**
- Verify server is running: `./scripts/start-api-server.sh`
- Check firewall: `sudo ufw allow 8080`
- Confirm IP address: `ip addr show`

**Webhook Not Received:**
- Verify webhook URL is accessible from Jetson (192.168.1.196)
- Check webhook server logs
- Test with manual POST from Jetson

**Out of Memory:**
- Reduce `--ctx-size` in startup script
- Process fewer pages per job
- Close other GPU applications

---

## Changelog

**v1.0.0** (2025-02-01)
- Initial API specification
- Async job processing with webhooks
- Support for PDF and image analysis
- JSON output with metadata extraction
- Image extraction with reference URLs
- Server IP: 192.168.1.196:8080

---

## Support

For issues or questions:
- Check server logs: `tail -f logs/api-server.log`
- Review documentation: `docs/qwen3vl-setup.md`
- Test endpoints: `curl http://192.168.1.196:8080/api/v1/health`
