#!/bin/bash
# VLM API Server Launcher
# Starts both llama-server and the API wrapper

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="/home/michael/src/llama.cpp/build/bin/llama-server"
MODEL_PATH="/home/michael/models/qwen3vl/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ_PATH="/home/michael/models/qwen3vl/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
API_SERVER="/home/michael/sandriverfishbot/api-server.py"
PORT=8080
LOG_DIR="/home/michael/sandriverfishbot/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}VLM API Server Launcher${NC}"
echo "================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$LLAMA_SERVER" ]; then
    echo -e "${RED}ERROR: llama-server not found at $LLAMA_SERVER${NC}"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$MMPROJ_PATH" ]; then
    echo -e "${RED}ERROR: Vision projection not found at $MMPROJ_PATH${NC}"
    exit 1
fi

if [ ! -f "$API_SERVER" ]; then
    echo -e "${RED}ERROR: API server not found at $API_SERVER${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Setup Python environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Install Python dependencies if needed
echo "Checking Python dependencies..."
source "$VENV_DIR/bin/activate"
pip show flask flask-cors requests 2>/dev/null >/dev/null || {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install flask flask-cors requests
}
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Kill any existing processes on port 8080
echo "Cleaning up existing processes..."
lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2

# Start llama-server in background
echo -e "${GREEN}Starting llama-server...${NC}"
$LLAMA_SERVER \
    -m "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --port $PORT \
    -ngl 99 \
    --ctx-size 8192 \
    --threads 8 \
    --chat-template chatml \
    > "$LOG_DIR/llama-server.log" 2>&1 &

LLAMA_PID=$!
echo "  PID: $LLAMA_PID"
echo "  Log: $LOG_DIR/llama-server.log"
echo ""

# Wait for llama-server to be ready
echo "Waiting for llama-server to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        echo -e "${GREEN}✓ llama-server ready${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Check if llama-server started successfully
if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo -e "${RED}ERROR: llama-server failed to start${NC}"
    echo "Check log: $LOG_DIR/llama-server.log"
    exit 1
fi

# Start API server
echo -e "${GREEN}Starting API server...${NC}"
export WEBHOOK_SECRET="${WEBHOOK_SECRET:-vlm-webhook-secret-2025}"
source "$VENV_DIR/bin/activate"
python "$API_SERVER" > "$LOG_DIR/api-server.log" 2>&1 &

API_PID=$!
echo "  PID: $API_PID"
echo "  Log: $LOG_DIR/api-server.log"
echo "  Webhook Secret: $WEBHOOK_SECRET"
echo ""

# Wait for API server
echo "Waiting for API server to initialize..."
for i in {1..10}; do
    if curl -s http://localhost:$PORT/api/v1/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ API server ready${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""
echo ""

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}VLM API Server Running!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Server IP: 192.168.1.196"
echo "Port: $PORT"
echo ""
echo "Endpoints:"
echo "  Health Check:  http://192.168.1.196:$PORT/api/v1/health"
echo "  Submit Job:    http://192.168.1.196:$PORT/api/v1/jobs"
echo "  Check Status:  http://192.168.1.196:$PORT/api/v1/jobs/{job_id}"
echo "  Get Results:   http://192.168.1.196:$PORT/api/v1/jobs/{job_id}/result"
echo "  Queue Status:  http://192.168.1.196:$PORT/api/v1/queue"
echo "  WebUI:         http://192.168.1.196:$PORT"
echo ""
echo "Logs:"
echo "  llama-server:  tail -f $LOG_DIR/llama-server.log"
echo "  API server:    tail -f $LOG_DIR/api-server.log"
echo ""
echo "Process IDs:"
echo "  llama-server:  $LLAMA_PID"
echo "  API server:    $API_PID"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
echo ""

# Save PIDs for stop script
echo "$LLAMA_PID $API_PID" > /tmp/vlm-server.pids

# Wait for interrupt
trap "echo ''; echo -e '${YELLOW}Stopping servers...${NC}'; kill $API_PID $LLAMA_PID 2>/dev/null; exit 0" INT
wait
