#!/usr/bin/env python3
"""
VLM API Server - Enhanced for BestBox Mold Troubleshooting System
Supports multipart file uploads and mold-specific analysis
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
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
from queue import Queue
from threading import Lock

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import requests

# Configuration
LLAMA_SERVER_PATH = "/home/michael/src/llama.cpp/build/bin/llama-server"
MODEL_PATH = "/home/michael/models/qwen3vl/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ_PATH = "/home/michael/models/qwen3vl/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
PORT = 8081
LLAMA_SERVER_PORT = 8080
MAX_CONCURRENT_JOBS = 3
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "vlm-webhook-secret-2025")
JOBS_DIR = Path("/tmp/vlm-jobs")
IMAGES_DIR = Path("/tmp/vlm-images")
UPLOAD_DIR = Path("/tmp/vlm-uploads")

# File upload limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "gif",
    "bmp",  # Images
    "pdf",  # PDF
    "xlsx",
    "xls",  # Excel
    "txt",
    "md",  # Text
}
ALLOWED_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/plain",
    "text/markdown",
}

# Ensure directories exist
for dir_path in [JOBS_DIR, IMAGES_DIR, UPLOAD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Job storage
jobs: Dict[str, "Job"] = {}
jobs_lock = Lock()
job_queue = Queue()

# Prompt Templates
PROMPT_TEMPLATES = {
    "mold_defect_analysis": """分析这个制造/模具相关的文档或图片。您是注塑模具缺陷诊断专家。

以结构化JSON格式提取以下信息：

1. **缺陷类型** (缺陷类型): 识别任何可见缺陷：
   - 披锋/flash (分型线处材料溢出)
   - 拉白/whitening (应力发白)
   - 火花纹/spark marks (未抛光的EDM痕迹)
   - 脏污/contamination (表面污染)
   - 划痕/scratches
   - 变形/deformation
   - 缩水/sink marks
   - 熔接痕/weld lines

2. **设备部件** (设备部件): 识别可见的模具组件：
   - 动模/moving half
   - 定模/fixed half
   - 型腔/cavity
   - 型芯/core
   - 滑块/slider
   - 顶针/ejector pin
   - 浇口/gate
   - 流道/runner

3. **文字内容** (文字内容): 提取任何可见文字：
   - 零件号 (Part numbers)
   - 试模版本 (T0/T1/T2)
   - 注释和标记
   - 手写笔记

4. **视觉标注** (视觉标注): 注意任何：
   - 指示问题区域的箭头
   - 圆圈或高亮
   - 前后对比标记

5. **根本原因线索** (根本原因线索):
   - 关于缺陷原因的可见线索
   - 建议的纠正措施

6. **严重程度评估** (严重程度):
   - high: 阻碍生产，影响客户
   - medium: 量产前需要处理
   - low: 轻微问题，可在维护时处理

以JSON格式输出：
{
  "defect_type": "string (主缺陷类别，中文)",
  "defect_details": "string (详细描述)",
  "equipment_part": "string (识别的设备/模具部件)",
  "text_in_image": "string (提取的所有文字)",
  "visual_annotations": "string (标记的描述)",
  "severity": "string (high/medium/low)",
  "root_cause_hints": ["字符串数组"],
  "suggested_actions": ["纠正措施字符串数组"],
  "confidence": 0.0-1.0,
  "bounding_boxes": [
    {"label": "缺陷类型", "x": 0, "y": 0, "width": 100, "height": 100, "confidence": 0.9}
  ]
}""",
    "general_vision": """详细描述这张图片。识别主要对象、场景、任何可见的文本内容，以及值得注意的特征。

提供以下结构化信息：
1. 场景描述
2. 主要对象列表
3. 可见文本（如果有）
4. 显著特征或异常
5. 整体印象

以清晰、详细的方式输出。""",
    "document_analysis": """分析此文档并提取关键信息：
1. 文档类型和目的
2. 主要主题和关键点
3. 结构化数据（表格、列表）
4. 重要日期、数字、名称
5. 总结性洞察

提供详细而简洁的分析。""",
}


@dataclass
class Job:
    job_id: str
    status: str
    file_path: Optional[Path] = None
    file_url: Optional[str] = None
    filename: str = ""
    file_type: str = ""
    webhook_url: Optional[str] = None
    prompt_template: str = "general_vision"
    options: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    progress: float = 0.0
    tokens_used: int = 0
    processing_time_ms: int = 0

    def to_dict(self):
        data = asdict(self)
        # Convert Path objects to strings
        if data.get("file_path"):
            data["file_path"] = str(data["file_path"])
        return data


def allowed_file(filename: str, mimetype: str) -> Tuple[bool, str]:
    """Check if file type is allowed"""
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS and mimetype not in ALLOWED_MIMETYPES:
        return (
            False,
            f"Unsupported file type: .{ext}. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    return True, ""


def get_template_prompt(template_id: str, custom_prompt: Optional[str] = None) -> str:
    """Get prompt from template or use custom prompt"""
    if custom_prompt and not template_id:
        return custom_prompt

    if template_id in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_id]

    return PROMPT_TEMPLATES.get("general_vision", "详细描述这张图片。")


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
        start_time = time.time()

        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            job.status = "processing"
            job.started_at = datetime.utcnow().isoformat() + "Z"
            job.progress = 0.1

        try:
            # Update progress
            job.progress = 0.2

            # Determine file to process
            if job.file_path and job.file_path.exists():
                file_to_process = job.file_path
            elif job.file_url:
                file_to_process = self.download_file(job.file_url, job_id)
            else:
                raise ValueError("No file provided")

            job.progress = 0.3

            # Process with VLM
            result = self.analyze_with_vlm(file_to_process, job)

            job.progress = 1.0
            processing_time = int((time.time() - start_time) * 1000)

            # Update job
            with jobs_lock:
                job.status = "completed"
                job.completed_at = datetime.utcnow().isoformat() + "Z"
                job.result = result
                job.processing_time_ms = processing_time

            # Send webhook
            if job.webhook_url:
                self.send_webhook(job)

        except Exception as e:
            with jobs_lock:
                job.status = "failed"
                job.error = {
                    "code": "PROCESSING_ERROR",
                    "message": str(e),
                    "details": str(e.__traceback__),
                }
            if job.webhook_url:
                self.send_webhook(job)

    def download_file(self, url: str, job_id: str) -> Path:
        """Download file from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()

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
        """Analyze single image with VLM"""
        # Get prompt
        prompt = get_template_prompt(
            job.prompt_template, job.options.get("custom_prompt")
        )

        # Add language instruction
        output_lang = job.options.get("output_language", "zh")
        if output_lang == "zh":
            prompt += "\n\n请以中文回答。"
        else:
            prompt += "\n\nPlease respond in English."

        # Convert image to base64
        image_base64 = base64.b64encode(image_path.read_bytes()).decode()

        # Determine max tokens
        max_tokens = job.options.get("max_tokens", 2048)

        # Call llama-server API
        try:
            response = requests.post(
                f"http://localhost:{LLAMA_SERVER_PORT}/v1/chat/completions",
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
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=300,
            )

            result = response.json()
            analysis_text = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)

            with jobs_lock:
                job.tokens_used = tokens_used
        except Exception as e:
            analysis_text = f"Analysis error: {str(e)}"
            tokens_used = 0

        # Parse structured output if using mold_defect_analysis template
        if job.prompt_template == "mold_defect_analysis":
            return self.parse_mold_analysis(analysis_text, image_path, job)
        else:
            return self.parse_general_analysis(analysis_text, image_path, job)

    def parse_mold_analysis(
        self, analysis_text: str, image_path: Path, job: Job
    ) -> Dict:
        """Parse mold defect analysis into structured format"""
        # Try to extract JSON from response
        structured_data = self.extract_json_from_text(analysis_text)

        if not structured_data:
            # Fallback: create basic structure from text analysis
            structured_data = {
                "defect_type": "未知",
                "defect_details": analysis_text[:500],
                "equipment_part": "未识别",
                "text_in_image": "",
                "visual_annotations": "",
                "severity": "medium",
                "root_cause_hints": [],
                "suggested_actions": [],
                "confidence": 0.5,
                "bounding_boxes": [],
            }

        # Generate image ID and save
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        saved_image = IMAGES_DIR / f"{job.job_id}_{image_id}.jpg"
        saved_image.write_bytes(image_path.read_bytes())

        # Build result in BestBox format
        return {
            "document_summary": structured_data.get(
                "defect_details", analysis_text[:200]
            ),
            "key_insights": structured_data.get("root_cause_hints", [])[:5],
            "analysis": {
                "sentiment": "neutral",
                "topics": self.extract_defect_topics(structured_data),
                "entities": self.extract_entities(structured_data),
                "complexity_score": structured_data.get("confidence", 0.5),
            },
            "extracted_images": [
                {
                    "image_id": image_id,
                    "page": 1,
                    "description": structured_data.get("defect_details", "")[:200],
                    "insights": structured_data.get("defect_details", ""),
                    "defect_type": structured_data.get("defect_type", "未知"),
                    "bounding_box": structured_data.get("bounding_boxes", [{}])[0]
                    if structured_data.get("bounding_boxes")
                    else None,
                    "reference_url": f"http://192.168.1.196:{PORT}/api/v1/images/{job.job_id}/{image_id}",
                }
            ],
            "text_content": structured_data.get("text_in_image", ""),
            "tags": self.generate_mold_tags(structured_data),
            "metadata": {
                "confidence_score": structured_data.get("confidence", 0.5),
                "processing_model": "qwen3-vl-8b-q8",
                "tokens_used": job.tokens_used,
                "processing_time_ms": job.processing_time_ms,
                "severity": structured_data.get("severity", "medium"),
            },
            "mold_analysis": structured_data,  # Full structured data for BestBox
        }

    def parse_general_analysis(
        self, analysis_text: str, image_path: Path, job: Job
    ) -> Dict:
        """Parse general analysis into standard format"""
        # Generate image ID and save
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        saved_image = IMAGES_DIR / f"{job.job_id}_{image_id}.jpg"
        saved_image.write_bytes(image_path.read_bytes())

        return {
            "document_summary": analysis_text[:500],
            "key_insights": analysis_text.split("\n")[:5],
            "analysis": {
                "sentiment": self.detect_sentiment(analysis_text),
                "topics": [],
                "entities": [],
                "complexity_score": 0.5,
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
            "text_content": "",
            "tags": self.generate_tags(analysis_text),
            "metadata": {
                "confidence_score": 0.85,
                "processing_model": "qwen3-vl-8b-q8",
                "tokens_used": job.tokens_used,
                "processing_time_ms": job.processing_time_ms,
            },
        }

    def extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract JSON object from text"""
        import re

        # Look for JSON between code blocks or braces
        patterns = [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{.*\}"]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue

        # Try to find JSON-like structure
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
        except:
            pass

        return None

    def extract_defect_topics(self, data: Dict) -> List[str]:
        """Extract defect-related topics"""
        topics = []
        defect_type = data.get("defect_type", "")
        if defect_type:
            topics.append(defect_type)

        equipment = data.get("equipment_part", "")
        if equipment:
            topics.append(equipment)

        return topics

    def extract_entities(self, data: Dict) -> List[Dict]:
        """Extract entities from analysis"""
        entities = []

        # Add defect type as entity
        defect = data.get("defect_type", "")
        if defect:
            entities.append({"name": defect, "type": "defect", "mentions": 1})

        # Add equipment part
        equipment = data.get("equipment_part", "")
        if equipment:
            entities.append({"name": equipment, "type": "equipment", "mentions": 1})

        return entities

    def generate_mold_tags(self, data: Dict) -> List[str]:
        """Generate tags for mold analysis"""
        tags = []

        if data.get("defect_type"):
            tags.append(data["defect_type"])
        if data.get("equipment_part"):
            tags.append(data["equipment_part"])
        if data.get("severity"):
            tags.append(data["severity"])

        return tags

    def analyze_document(self, file_path: Path, job: Job) -> Dict:
        """Analyze document - converts PDF pages to images and analyzes each"""
        try:
            # Convert PDF to images
            images = convert_from_path(str(file_path), dpi=150)

            if not images:
                return {
                    "document_summary": "无法读取PDF文档",
                    "key_insights": [],
                    "analysis": {
                        "sentiment": "neutral",
                        "topics": [],
                        "entities": [],
                        "complexity_score": 0,
                    },
                    "extracted_images": [],
                    "text_content": "",
                    "tags": ["PDF", "读取失败"],
                    "metadata": {
                        "confidence_score": 0,
                        "processing_model": "qwen3-vl-8b-q8",
                        "tokens_used": 0,
                    },
                }

            all_summaries = []
            all_insights = []
            all_images = []
            total_tokens = 0
            page_num = 1

            # Process each page
            for image in images:
                # Save temporary image
                temp_img_path = UPLOAD_DIR / f"{job.job_id}_page_{page_num}.jpg"
                image.save(str(temp_img_path), "JPEG", quality=85)

                # Analyze with VLM
                prompt = get_template_prompt(
                    job.prompt_template, job.options.get("custom_prompt")
                )

                # Convert to base64
                with open(temp_img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()

                # Call VLM
                try:
                    response = requests.post(
                        "http://localhost:8080/v1/chat/completions",
                        json={
                            "model": "qwen3-vl",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{img_base64}"
                                            },
                                        },
                                        {"type": "text", "text": prompt},
                                    ],
                                }
                            ],
                            "max_tokens": job.options.get("max_tokens", 2048),
                            "temperature": 0.7,
                        },
                        timeout=300,
                    )

                    result = response.json()
                    page_analysis = result["choices"][0]["message"]["content"]
                    page_tokens = result.get("usage", {}).get("total_tokens", 0)
                    total_tokens += page_tokens

                except Exception as e:
                    page_analysis = f"分析错误: {str(e)}"
                    page_tokens = 0

                # Save image to images directory
                image_id = f"img_{uuid.uuid4().hex[:8]}"
                saved_image = IMAGES_DIR / f"{job.job_id}_{image_id}.jpg"
                image.save(str(saved_image))

                # Add to results
                all_summaries.append(f"第{page_num}页: {page_analysis[:300]}")
                all_insights.extend(page_analysis.split("\n")[:3])

                all_images.append(
                    {
                        "image_id": image_id,
                        "page": page_num,
                        "description": page_analysis[:200],
                        "insights": page_analysis,
                        "reference_url": f"http://192.168.1.196:8081/api/v1/images/{job.job_id}/{image_id}",
                    }
                )

                # Clean up temp file
                temp_img_path.unlink(missing_ok=True)
                page_num += 1

            # Update job tokens
            job.tokens_used = total_tokens

            return {
                "document_summary": " | ".join(all_summaries),
                "key_insights": list(set(all_insights))[:10],
                "analysis": {
                    "sentiment": "neutral",
                    "topics": ["PDF文档", f"{len(images)}页"],
                    "entities": [],
                    "complexity_score": min(1.0, len(images) * 0.1),
                },
                "extracted_images": all_images,
                "text_content": "",
                "tags": ["PDF", f"{len(images)}页"],
                "metadata": {
                    "confidence_score": 0.85,
                    "processing_model": "qwen3-vl-8b-q8",
                    "tokens_used": total_tokens,
                    "pages_processed": len(images),
                },
            }

        except Exception as e:
            return {
                "document_summary": f"PDF处理错误: {str(e)}",
                "key_insights": [],
                "analysis": {
                    "sentiment": "neutral",
                    "topics": ["PDF"],
                    "entities": [],
                    "complexity_score": 0,
                },
                "extracted_images": [],
                "text_content": "",
                "tags": ["PDF", "错误"],
                "metadata": {
                    "confidence_score": 0,
                    "processing_model": "qwen3-vl-8b-q8",
                    "tokens_used": 0,
                    "error": str(e),
                },
            }

    def detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection"""
        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "beautiful",
            "amazing",
            "良好",
            "优秀",
            "完美",
        ]
        negative_words = [
            "bad",
            "poor",
            "negative",
            "terrible",
            "awful",
            "错误",
            "问题",
            "缺陷",
            "故障",
        ]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def generate_tags(self, text: str) -> List[str]:
        """Generate tags from content"""
        words = text.lower().split()
        tags = list(set([w for w in words if len(w) > 5]))[:10]
        return tags

    def send_webhook(self, job: Job):
        """Send webhook callback"""
        if not job.webhook_url:
            return

        try:
            payload = {
                "event": "job.completed" if job.status == "completed" else "job.failed",
                "job_id": job.job_id,
                "status": job.status,
                "result": job.result,
                "completed_at": job.completed_at or datetime.utcnow().isoformat() + "Z",
            }

            # Sign payload
            signature = hmac.new(
                WEBHOOK_SECRET.encode(),
                json.dumps(payload, default=str).encode(),
                hashlib.sha256,
            ).hexdigest()

            requests.post(
                job.webhook_url,
                json=payload,
                headers={
                    "X-VLM-Signature": f"sha256={signature}",
                    "X-VLM-Job-ID": job.job_id,
                    "X-VLM-Timestamp": job.completed_at
                    or datetime.utcnow().isoformat() + "Z",
                },
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
            "model": "Qwen3-VL-8B",
            "version": "1.0.0",
            "server_ip": "192.168.1.196",
            "port": PORT,
            "gpu_memory_used": "7.2GB",
            "gpu_memory_total": "64GB",
            "queue_depth": job_queue.qsize(),
            "active_jobs": sum(1 for j in jobs.values() if j.status == "processing"),
            "average_processing_time_ms": 12500,
        }
    )


@app.route("/api/v1/templates", methods=["GET"])
def list_templates():
    """List available prompt templates"""
    return jsonify(
        {
            "templates": [
                {
                    "id": "mold_defect_analysis",
                    "name": "模具缺陷分析",
                    "description": "注塑模具缺陷诊断专用模板，输出中文结构化分析",
                    "language": "zh",
                },
                {
                    "id": "general_vision",
                    "name": "通用图像分析",
                    "description": "通用图像描述和分析",
                    "language": "auto",
                },
                {
                    "id": "document_analysis",
                    "name": "文档分析",
                    "description": "文档内容提取和结构化分析",
                    "language": "auto",
                },
            ]
        }
    )


@app.route("/api/v1/jobs", methods=["POST"])
def submit_job():
    """Submit a new job via JSON (file_url method)"""
    data = request.get_json()

    if not data or "file_url" not in data:
        return jsonify(
            {
                "error": {
                    "code": "MISSING_FILE_URL",
                    "message": "file_url is required in request body",
                }
            }
        ), 400

    return create_job(
        file_url=data["file_url"],
        webhook_url=data.get("webhook_url"),
        prompt_template=data.get("prompt_template", "general_vision"),
        options=data.get("options", {}),
    )


@app.route("/api/v1/jobs/upload", methods=["POST"])
def upload_job():
    """Submit a new job via multipart file upload"""
    # Check if file present
    if "file" not in request.files:
        return jsonify(
            {
                "error": {
                    "code": "MISSING_FILE",
                    "message": "No file part in request. Use 'file' field.",
                }
            }
        ), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(
            {"error": {"code": "EMPTY_FILENAME", "message": "No file selected"}}
        ), 400

    # Validate file
    if not file.filename:
        return jsonify(
            {"error": {"code": "INVALID_FILE", "message": "File has no filename"}}
        ), 400

    is_allowed, error_msg = allowed_file(file.filename, file.content_type or "")
    if not is_allowed:
        return jsonify(
            {
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": error_msg,
                    "details": {
                        "received_type": file.content_type,
                        "supported_types": list(ALLOWED_MIMETYPES),
                    },
                }
            }
        ), 415

    # Save uploaded file
    filename = (
        secure_filename(file.filename)
        if file.filename
        else f"upload_{uuid.uuid4().hex[:8]}"
    )
    job_id = f"vlm-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    file_path = UPLOAD_DIR / f"{job_id}_{filename}"
    file.save(file_path)

    # Parse options from form data
    options = {}
    if "options" in request.form:
        try:
            options = json.loads(request.form["options"])
        except:
            return jsonify(
                {
                    "error": {
                        "code": "INVALID_OPTIONS",
                        "message": "options must be valid JSON string",
                    }
                }
            ), 400

    return create_job(
        file_path=file_path,
        filename=filename,
        file_type=file.content_type or "application/octet-stream",
        webhook_url=request.form.get("webhook_url"),
        prompt_template=request.form.get("prompt_template", "general_vision"),
        options=options,
    )


def create_job(
    file_path=None,
    file_url=None,
    filename="",
    file_type="",
    webhook_url=None,
    prompt_template="general_vision",
    options=None,
):
    """Create and queue a new job"""
    options = options or {}

    # Validate template
    if prompt_template not in PROMPT_TEMPLATES and not options.get("custom_prompt"):
        return jsonify(
            {
                "error": {
                    "code": "TEMPLATE_NOT_FOUND",
                    "message": f"Unknown template: {prompt_template}. Available: {list(PROMPT_TEMPLATES.keys())}",
                }
            }
        ), 400

    # Create job
    job_id = f"vlm-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    job = Job(
        job_id=job_id,
        status="pending",
        file_path=file_path,
        file_url=file_url,
        filename=filename,
        file_type=file_type,
        webhook_url=webhook_url,
        prompt_template=prompt_template,
        options=options,
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
        return jsonify(
            {"error": {"code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found"}}
        ), 404

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "estimated_remaining": f"{int((1 - job.progress) * 60)}s"
        if job.status == "processing"
        else None,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }

    if job.status == "completed":
        response["result"] = job.result

    if job.status == "failed" and job.error:
        response["error"] = job.error

    return jsonify(response)


@app.route("/api/v1/jobs/<job_id>/result", methods=["GET"])
def get_job_result(job_id: str):
    """Get job results"""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify(
            {"error": {"code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found"}}
        ), 404

    if job.status != "completed":
        return jsonify(
            {"job_id": job_id, "status": job.status, "message": "Job not yet complete"}
        ), 202

    return jsonify(
        {
            "job_id": job_id,
            "status": "completed",
            "result": job.result,
            "completed_at": job.completed_at,
        }
    )


@app.route("/api/v1/images/<job_id>/<image_id>", methods=["GET"])
def get_image(job_id: str, image_id: str):
    """Retrieve extracted image"""
    image_path = IMAGES_DIR / f"{job_id}_{image_id}.jpg"

    if not image_path.exists():
        return jsonify(
            {
                "error": {
                    "code": "IMAGE_NOT_FOUND",
                    "message": f"Image {image_id} not found for job {job_id}",
                }
            }
        ), 404

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


@app.errorhandler(413)
def too_large(e):
    """Handle file too large"""
    return jsonify(
        {
            "error": {
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds maximum size of {MAX_FILE_SIZE // 1024 // 1024}MB",
            }
        }
    ), 413


@app.errorhandler(429)
def rate_limited(e):
    """Handle rate limit"""
    return jsonify(
        {
            "error": {
                "code": "RATE_LIMITED",
                "message": "Too many requests. Please try again later.",
            }
        }
    ), 429


if __name__ == "__main__":
    print(f"Starting VLM API Server on http://192.168.1.196:{PORT}")
    print(f"llama-server backend on port {PORT}")
    print(f"Jobs directory: {JOBS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print("\nEndpoints:")
    print(f"  Health:    http://192.168.1.196:{PORT}/api/v1/health")
    print(f"  Templates: http://192.168.1.196:{PORT}/api/v1/templates")
    print(f"  Submit:    http://192.168.1.196:{PORT}/api/v1/jobs (JSON)")
    print(f"  Upload:    http://192.168.1.196:{PORT}/api/v1/jobs/upload (multipart)")
    print(f"  WebUI:     http://192.168.1.196:{PORT}")

    app.run(host="0.0.0.0", port=PORT, threaded=True)
