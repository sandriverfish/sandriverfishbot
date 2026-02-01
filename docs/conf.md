# Setup Jetson AGX Orin with OpenClaw

https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/

## Procedures
```bash
docker run --rm -it \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --name=vllm \
  -v $HOME/data/models/huggingface:/root/.cache/huggingface \
  -v $HOME/data/vllm_cache:/root/.cache/vllm \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
```

This computer was reflashed and upgrade JetPack from 5.x to now 6.2.1, before flash I copied the downloaded docker image vllm, please check the connected harddrive, mount it and restore to this computer.

```
sudo usermod -aG docker $USER
newgrp docker            # or log out and log back in
docker run hello-world   # should work without sudo

docker run --rm -it \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --name=vllm \
  -v $HOME/data/models/huggingface:/root/.cache/huggingface \
  -v $HOME/data/vllm_cache:/root/.cache/vllm \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin


sudo docker stop vllm
sudo docker rm vllm
# then run your original command
```

## Models

https://huggingface.co/collections/Qwen/qwen3-vl

pip3 install --user -U huggingface_hub tqdm
