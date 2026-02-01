#!/bin/bash

# Stop the Qwen3-VL llama-server process running on port 8080

PORT=8080
PID=""

# Try to find PID using lsof first
echo "Looking for llama-server process on port $PORT..."

if command -v lsof &> /dev/null; then
    PID=$(lsof -t -i :$PORT 2>/dev/null)
fi

# Fallback to pgrep if lsof not available or no PID found
if [ -z "$PID" ]; then
    PID=$(pgrep -f "llama-server.*Qwen3VL" 2>/dev/null)
fi

# Check if we found a PID
if [ -z "$PID" ]; then
    echo "No llama-server process found on port $PORT or matching 'llama-server.*Qwen3VL'"
    exit 0
fi

echo "Found llama-server process (PID: $PID). Stopping..."

# Send SIGTERM first
kill -TERM $PID 2>/dev/null

# Wait briefly for graceful shutdown
sleep 2

# Check if process is still running
if kill -0 $PID 2>/dev/null; then
    echo "Process still running, sending SIGKILL..."
    kill -KILL $PID 2>/dev/null
fi

# Verify process stopped
sleep 1
if ! kill -0 $PID 2>/dev/null; then
    echo "✓ llama-server process stopped successfully (PID: $PID)"
else
    echo "Failed to stop process (PID: $PID)"
    exit 1
fi

exit 0
