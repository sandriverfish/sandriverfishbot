# Qwen3-VL 8B Vision Model Setup

## Quick Start

1. **Start the server:**
   ```bash
   ./scripts/start-qwen3vl-server.sh
   ```

2. **Access the WebUI:**
   Open http://localhost:8080 in your browser

3. **Stop the server:**
   - Press `Ctrl+C` in the terminal running the server, or
   - Run: `./scripts/stop-qwen3vl-server.sh`

## What's Included

- **Model:** Qwen3-VL-8B-Instruct-Q8_0.gguf (8-bit quantized)
- **Vision projection file** for image understanding
- **llama-server** with OpenAI-compatible API
- **Built-in Svelte-based WebUI**

## API Usage

### Available Endpoints

- **Chat Completions:** `POST /v1/chat/completions`
- **Models List:** `GET /v1/models`

### Example Chat Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Testing Vision

Test the vision capabilities with these steps:

1. Open the WebUI at http://localhost:8080
2. Click the attachment/clip icon to upload an image
3. Ask questions about the image in the chat input

The model will analyze the image and respond to your questions about it.

## Configuration Details

| Setting | Value |
|---------|-------|
| Port | 8080 |
| GPU Layers | 99 (Jetson AGX Orin optimized) |
| Context Size | 8192 |
| Chat Template | chatml |
