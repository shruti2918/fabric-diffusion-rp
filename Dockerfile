FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.21.0 \
    transformers==4.35.0 \
    accelerate \
    Pillow \
    huggingface_hub

RUN git clone https://github.com/humansensinglab/fabric-diffusion /app/fabric-diffusion

COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
