mkdir whisper_realtime_server_client && cd whisper_realtime_server_client

# Create requirements.txt
cat > requirements.txt <<'EOF'
fastapi==0.112.2
uvicorn[standard]==0.30.6
websockets==13.0
numpy==2.1.1
sounddevice==0.4.7
faster-whisper==1.0.3
webrtcvad==2.0.10
pydantic==2.9.2
EOF

# Create server.py
cat > server.py <<'EOF'
<PASTE SERVER CODE FROM ABOVE HERE>
EOF

# Create client.py
cat > client.py <<'EOF'
<PASTE CLIENT CODE FROM ABOVE HERE>
EOF

# Create Dockerfile
cat > Dockerfile <<'EOF'
# CUDA runtime base (adjust tag to match your host driver)
FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY server.py /app/server.py

EXPOSE 8000
ENV WHISPER_MODEL=large-v3
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create README.md
cat > README.md <<'EOF'
<PASTE README CONTENT FROM ABOVE HERE>
EOF
docker build -t whisper-rt .
docker run --gpus all -p 8000:8000 whisper-rt

for cpu
# Build the image
docker build -t whisper-rt-cpu .

# Run the container
docker run -p 8000:8000 whisper-rt-cpu