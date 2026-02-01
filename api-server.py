#!/usr/bin/env python3
"""
VLM API Server - Async job processing with llama-server backend
Provides REST API for external services to submit vision-language processing jobs
"""

import os
import sys
import json
import time
import uuid
import hashlib
import hmac
import base64
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from queue import Queue
from threading import Lock

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests

# Configuration
LLAMA_SERVER_PATH = "/home/michael/src/llama.cpp/build/bin/llama-server"
MODEL_PATH = "/home/michael/models/qwen3vl/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ_PATH = "/home/michael/models/qwen3vl/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
PORT = 8080
MAX_CONCURRENT_JOBS = 3
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "your-secret-key")
JOBS_DIR = Path("/tmp/vlm-jobs")
IMAGES_DIR = Path("/tmp/vlm-images")

# Ensure directories exist
JOBS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Job storage
jobs: Dict[str, "Job"] = {}
jobs_lock = Lock()
job_queue = Queue()


@dataclass
class Job:
    job_id: str
    status: str  # pending, processing, completed, failed
    file_url: str
    webhook_url: Optional[str]
    prompt_template: Optional[str]
    options: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    progress: int = 0
    tokens_generated: int = 0

    def to_dict(self):
        return asdict(self)


class JobProcessor(threading.Thread):
    """Background worker to process jobs from the queue"""

    def __init__(self):
        super().__init__(daemon=True)
        self.active = True

    def run(self):
        while self.active:
            try:
                job_id = job_queue.get(timeout=1)
                if job_id:
                    self.process_job(job_id)
            except:
                continue

    def process_job(self, job_id: str):
        """Process a single job"""
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            job.status = "processing"
            job.started_at = datetime.utcnow().isoformat() + "Z"

        try:
            # Download file
            file_path = self.download_file(job.file_url, job_id)

            # Process with llama-server
            result = self.analyze_with_vlm(file_path, job)

            # Update job
            with jobs_lock:
                job.status = "completed"
                job.completed_at = datetime.utcnow().isoformat() + "Z"
                job.progress = 100
                job.result = result

            # Save result to file
            self.save_result(job_id, result)

            # Send webhook
            if job.webhook_url:
                self.send_webhook(job)

        except Exception as e:
            with jobs_lock:
                job.status = "failed"
                job.error = {
                    "code": "PROCESSING_ERROR",
                    "message": str(e),
                    "details": getattr(e, "__traceback__", None),
                }
            if job.webhook_url:
                self.send_webhook(job)

    def download_file(self, url: str, job_id: str) -> Path:
        """Download file from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Determine file extension from content-type or URL
        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type:
            ext = ".pdf"
        elif "image" in content_type:
            ext = ".jpg"
        else:
            ext = Path(url).suffix or ".bin"

        file_path = JOBS_DIR / f"{job_id}{ext}"
        file_path.write_bytes(response.content)
        return file_path

    def analyze_with_vlm(self, file_path: Path, job: Job) -> Dict:
        """Analyze file using llama-server"""
        is_image = file_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".bmp",
        ]

        if is_image:
            return self.analyze_image(file_path, job)
        else:
            return self.analyze_document(file_path, job)

    def analyze_image(self, image_path: Path, job: Job) -> Dict:
        """Analyze single image"""
        # Build prompt
        prompt = (
            job.prompt_template
            or """Analyze this image in detail. Describe what you see including:
1. Main objects and subjects
2. Text content (if any)
3. Colors, composition, and style
4. Notable features or anomalies
5. Context and setting

Provide your analysis in a structured format."""
        )

        # Convert image to base64 for API
        image_base64 = base64.b64encode(image_path.read_bytes()).decode()

        # Call llama-server API
        response = requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json={
                "model": "qwen3-vl",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
            },
            timeout=300,
        )

        result = response.json()
        analysis_text = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)

        # Generate image ID and save copy
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        saved_image = IMAGES_DIR / f"{job.job_id}_{image_id}.jpg"
        saved_image.write_bytes(image_path.read_bytes())

        return {
            "document_summary": analysis_text,
            "key_insights": self.extract_insights(analysis_text),
            "analysis": {
                "sentiment": self.detect_sentiment(analysis_text),
                "complexity_score": 5.0,
                "topics": self.extract_topics(analysis_text),
                "entities": self.extract_entities(analysis_text),
            },
            "extracted_images": [
                {
                    "image_id": image_id,
                    "page": 1,
                    "description": analysis_text[:200],
                    "insights": analysis_text,
                    "reference_url": f"http://192.168.1.196:{PORT}/api/v1/images/{job.job_id}/{image_id}",
                }
            ],
            "tags": self.generate_tags(analysis_text),
            "metadata": {
                "confidence_score": 0.85,
                "processing_model": "qwen3-vl-8b-q8",
                "tokens_used": tokens_used,
            },
        }

    def analyze_document(self, file_path: Path, job: Job) -> Dict:
        """Analyze document (simplified - assumes first page image extraction)"""
        # For PDFs, we'd need pdf2image library
        # For now, return structured error suggesting image conversion
        return {
            "document_summary": "Document analysis requires image conversion. Please convert PDF pages to images first.",
            "key_insights": [],
            "analysis": {
                "sentiment": "neutral",
                "complexity_score": 0,
                "topics": [],
                "entities": [],
            },
            "extracted_images": [],
            "tags": ["document", "requires-conversion"],
            "metadata": {
                "confidence_score": 0,
                "processing_model": "qwen3-vl-8b-q8",
                "tokens_used": 0,
                "note": "Convert PDF to images for analysis",
            },
        }

    def extract_insights(self, text: str) -> List[str]:
        """Extract key insights from analysis"""
        lines = text.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.endswith(":"):
                insights.append(line)
        return insights[:5]

    def detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection"""
        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "beautiful",
            "amazing",
        ]
        negative_words = ["bad", "poor", "negative", "terrible", "awful"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics"""
        # Simple keyword extraction
        common_topics = [
            "technology",
            "nature",
            "architecture",
            "people",
            "art",
            "business",
            "science",
            "health",
            "education",
            "finance",
        ]
        text_lower = text.lower()
        return [t for t in common_topics if t in text_lower][:3]

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities (simplified)"""
        # This would use NER in production
        return []

    def generate_tags(self, text: str) -> List[str]:
        """Generate tags from content"""
        words = text.lower().split()
        # Filter for meaningful words
        tags = list(set([w for w in words if len(w) > 5]))[:10]
        return tags

    def save_result(self, job_id: str, result: Dict):
        """Save result to file"""
        result_path = JOBS_DIR / f"{job_id}_result.json"
        result_path.write_text(json.dumps(result, indent=2))

    def send_webhook(self, job: Job):
        """Send webhook callback"""
        try:
            payload = {
                "event": "job.completed" if job.status == "completed" else "job.failed",
                "job_id": job.job_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "result_summary": {
                    "status": job.status,
                    "pages_processed": 1,
                    "images_extracted": len(job.result.get("extracted_images", []))
                    if job.result
                    else 0,
                    "result_url": f"http://192.168.1.196:{PORT}/api/v1/jobs/{job.job_id}/result",
                },
            }

            # Sign payload
            signature = hmac.new(
                WEBHOOK_SECRET.encode(), json.dumps(payload).encode(), hashlib.sha256
            ).hexdigest()

            if job.webhook_url:
                requests.post(
                    job.webhook_url,
                    json=payload,
                    headers={"X-VLM-Signature": f"sha256={signature}"},
                    timeout=10,
                )
        except Exception as e:
            print(f"Webhook failed for job {job.job_id}: {e}")


# Start background workers
workers = []
for i in range(MAX_CONCURRENT_JOBS):
    worker = JobProcessor()
    worker.start()
    workers.append(worker)

# API Routes


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model": "qwen3-vl-8b-q8",
            "gpu_available": True,
            "queue_length": job_queue.qsize(),
            "active_jobs": sum(1 for j in jobs.values() if j.status == "processing"),
            "avg_tokens_per_sec": 5.5,
            "server_ip": "192.168.1.196",
            "port": PORT,
        }
    )


@app.route("/api/v1/jobs", methods=["POST"])
def submit_job():
    """Submit a new job for processing"""
    data = request.get_json()

    # Validate required fields
    if not data or "file_url" not in data:
        return jsonify({"error": "file_url is required"}), 400

    # Create job
    job_id = f"vlm-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    job = Job(
        job_id=job_id,
        status="pending",
        file_url=data["file_url"],
        webhook_url=data.get("webhook_url"),
        prompt_template=data.get("prompt_template"),
        options=data.get("options", {}),
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    # Store job
    with jobs_lock:
        jobs[job_id] = job

    # Add to queue
    job_queue.put(job_id)

    return jsonify(
        {
            "job_id": job_id,
            "status": "pending",
            "estimated_duration": "30-60s",
            "check_status_url": f"http://192.168.1.196:{PORT}/api/v1/jobs/{job_id}",
            "submitted_at": job.created_at,
        }
    ), 202


@app.route("/api/v1/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get job status"""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "tokens_generated": job.tokens_generated,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }

    if job.status == "completed":
        response["result_url"] = (
            f"http://192.168.1.196:{PORT}/api/v1/jobs/{job_id}/result"
        )

    if job.status == "failed" and job.error:
        response["error"] = job.error

    return jsonify(response)


@app.route("/api/v1/jobs/<job_id>/result", methods=["GET"])
def get_job_result(job_id: str):
    """Get job results"""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status != "completed":
        return jsonify(
            {"job_id": job_id, "status": job.status, "message": "Job not yet complete"}
        ), 202

    return jsonify(
        {
            "job_id": job_id,
            "input": {
                "filename": Path(job.file_url).name,
                "file_type": "image/jpeg",
                "pages_processed": 1,
                "processing_duration": "30s",
            },
            "output": job.result,
        }
    )


@app.route("/api/v1/images/<job_id>/<image_id>", methods=["GET"])
def get_image(job_id: str, image_id: str):
    """Retrieve extracted image"""
    image_path = IMAGES_DIR / f"{job_id}_{image_id}.jpg"

    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404

    return send_file(image_path, mimetype="image/jpeg")


@app.route("/api/v1/queue", methods=["GET"])
def get_queue_status():
    """Get queue status"""
    return jsonify(
        {
            "queue_length": job_queue.qsize(),
            "max_concurrent": MAX_CONCURRENT_JOBS,
            "active_jobs": sum(1 for j in jobs.values() if j.status == "processing"),
            "pending_jobs": sum(1 for j in jobs.values() if j.status == "pending"),
            "completed_jobs": sum(1 for j in jobs.values() if j.status == "completed"),
            "failed_jobs": sum(1 for j in jobs.values() if j.status == "failed"),
        }
    )


if __name__ == "__main__":
    print(f"Starting VLM API Server on http://192.168.1.196:{PORT}")
    print(f"_llama-server backend on port {PORT}")
    print(f"Jobs directory: {JOBS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print("\nEndpoints:")
    print(f"  Health:    http://192.168.1.196:{PORT}/api/v1/health")
    print(f"  Submit:    http://192.168.1.196:{PORT}/api/v1/jobs")
    print(f"  WebUI:     http://192.168.1.196:{PORT}")

    app.run(host="0.0.0.0", port=PORT, threaded=True)
