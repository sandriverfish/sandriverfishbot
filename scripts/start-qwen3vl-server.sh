#!/usr/bin/env bash
# Qwen3-VL 8B Server Startup Script
# Launches llama-server with vision model support

set -e

# Configuration
LLAMA_SERVER="/home/michael/src/llama.cpp/build/bin/llama-server"
MODEL="/home/michael/models/qwen3vl/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ="/home/michael/models/qwen3vl/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
PORT=8080
GPU_LAYERS=99
CTX_SIZE=8192
THREADS=8
CHAT_TEMPLATE="chatml"

echo "========================================="
echo "Qwen3-VL 8B Server Startup Script"
echo "========================================="
echo ""
echo "Checking required files..."
echo ""

# Verify llama-server binary exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server binary not found: $LLAMA_SERVER"
    exit 1
fi
echo "✓ llama-server binary: $LLAMA_SERVER"

# Verify model file exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi
echo "✓ Model file: $MODEL"

# Verify mmproj file exists
if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: Vision projection file not found: $MMPROJ"
    exit 1
fi
echo "✓ Vision projection: $MMPROJ"

echo ""
echo "All required files found."
echo ""
echo "========================================="
echo "Starting llama-server with Qwen3-VL..."
echo "========================================="
echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  GPU Layers: $GPU_LAYERS"
echo "  Context Size: $CTX_SIZE"
echo "  Threads: $THREADS"
echo "  Chat Template: $CHAT_TEMPLATE"
echo ""
echo "WebUI: http://localhost:$PORT"
echo "API Endpoint: http://localhost:$PORT/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch llama-server, replacing the current process
exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --port "$PORT" \
    -ngl "$GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --chat-template "$CHAT_TEMPLATE"
