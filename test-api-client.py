#!/usr/bin/env python3
"""
Test client for VLM API Server
Demonstrates how external servers can integrate with the VLM
"""

import requests
import time
import sys

# Configuration
VLM_BASE_URL = "http://192.168.1.196:8080"
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{VLM_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server healthy")
            print(f"  Model: {data.get('model')}")
            print(f"  Queue: {data.get('queue_length')} jobs")
            print(f"  Active: {data.get('active_jobs')} jobs")
            print(f"  Speed: {data.get('avg_tokens_per_sec')} tokens/sec")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_submit_job():
    """Test job submission"""
    print("\nTesting job submission...")
    try:
        # Submit a test job (using a simple image)
        response = requests.post(
            f"{VLM_BASE_URL}/api/v1/jobs",
            json={
                "file_url": TEST_IMAGE_URL,
                "webhook_url": None,  # No webhook for testing
                "options": {
                    "extract_images": True,
                    "generate_summary": True,
                    "analysis_depth": "detailed",
                },
            },
            timeout=10,
        )

        if response.status_code == 202:
            data = response.json()
            job_id = data["job_id"]
            print(f"✓ Job submitted: {job_id}")
            print(f"  Status: {data['status']}")
            print(f"  Estimated: {data['estimated_duration']}")
            return job_id
        else:
            print(f"✗ Job submission failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Job submission error: {e}")
        return None


def poll_job_status(job_id, max_attempts=30):
    """Poll job until completion"""
    print(f"\nPolling job {job_id}...")

    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{VLM_BASE_URL}/api/v1/jobs/{job_id}", timeout=5)

            if response.status_code == 200:
                data = response.json()
                status = data["status"]
                progress = data.get("progress", 0)

                if status == "completed":
                    print(f"✓ Job completed!")
                    return True
                elif status == "failed":
                    print(f"✗ Job failed: {data.get('error')}")
                    return False
                else:
                    print(
                        f"  Status: {status} ({progress}%) - Attempt {attempt + 1}/{max_attempts}"
                    )
            else:
                print(f"  Error checking status: {response.status_code}")

        except Exception as e:
            print(f"  Polling error: {e}")

        time.sleep(5)

    print(f"✗ Timeout waiting for job completion")
    return False


def get_job_results(job_id):
    """Get job results"""
    print(f"\nGetting results for job {job_id}...")

    try:
        response = requests.get(
            f"{VLM_BASE_URL}/api/v1/jobs/{job_id}/result", timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            output = data.get("output", {})

            print(f"✓ Results retrieved")
            print(f"\nDocument Summary:")
            print(f"  {output.get('document_summary', 'N/A')[:200]}...")

            print(f"\nKey Insights:")
            for insight in output.get("key_insights", [])[:3]:
                print(f"  - {insight}")

            print(f"\nMetadata:")
            metadata = output.get("metadata", {})
            print(f"  Tokens used: {metadata.get('tokens_used', 'N/A')}")
            print(f"  Confidence: {metadata.get('confidence_score', 'N/A')}")

            images = output.get("extracted_images", [])
            print(f"\nExtracted Images: {len(images)}")
            for img in images:
                print(
                    f"  - {img.get('image_id')}: {img.get('description', 'N/A')[:50]}..."
                )

            return data
        else:
            print(f"✗ Failed to get results: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error getting results: {e}")
        return None


def test_queue_status():
    """Test queue status endpoint"""
    print("\nTesting queue status...")
    try:
        response = requests.get(f"{VLM_BASE_URL}/api/v1/queue", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Queue status retrieved")
            print(f"  Queue length: {data.get('queue_length')}")
            print(f"  Active jobs: {data.get('active_jobs')}")
            print(f"  Pending: {data.get('pending_jobs')}")
            print(f"  Completed: {data.get('completed_jobs')}")
            return True
        else:
            print(f"✗ Queue status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Queue status error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("VLM API Server Test Client")
    print(f"Server: {VLM_BASE_URL}")
    print("=" * 60)

    # Test 1: Health check
    if not test_health():
        print("\n✗ Server not available. Please start the server first:")
        print("  ./scripts/start-api-server.sh")
        sys.exit(1)

    # Test 2: Queue status
    test_queue_status()

    # Test 3: Submit job
    job_id = test_submit_job()
    if not job_id:
        print("\n✗ Cannot continue without job submission")
        sys.exit(1)

    # Test 4: Poll for completion
    if poll_job_status(job_id):
        # Test 5: Get results
        get_job_results(job_id)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nFor production use:")
    print("1. Use webhooks instead of polling")
    print("2. Implement proper error handling")
    print("3. Store results in your knowledge base")
    print("4. Monitor queue depth to avoid overload")


if __name__ == "__main__":
    main()
