# Sand River Fish Bot

A project for running vision-language models on NVIDIA Jetson AGX Orin with OpenClaw.

## Quick Start - Vision Model (Qwen3-VL 8B)

Run the Qwen3-VL 8B vision-language model with built-in WebUI:

```bash
./scripts/start-qwen3vl-server.sh
```

Then open http://localhost:8080 in your browser to chat with vision capabilities.

For detailed setup instructions, see [docs/qwen3vl-setup.md](docs/qwen3vl-setup.md).

## Documentation

- [Qwen3-VL 8B Vision Model Setup](docs/qwen3vl-setup.md) - Detailed guide for running the vision-language model
- [Jetson AGX Orin Setup](docs/conf.md) - Initial setup and configuration with OpenClaw

## Scripts

Available helper scripts in the `scripts/` directory:

- `start-qwen3vl-server.sh` - Start the Qwen3-VL vision model server
- `stop-qwen3vl-server.sh` - Stop the running server
