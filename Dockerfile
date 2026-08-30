# Multi-arch base image supporting linux/arm64
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone and compile llama.cpp with KleidiAI
RUN git clone --recursive https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CPU_KLEIDIAI=ON \
      -DGGML_NATIVE=ON \
      -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release -j$(nproc)

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080 8501

CMD ["bash", "-c", "./llama.cpp/build/bin/llama-server -m models/llama-3.2-1b-instruct-q4_k_m.gguf --port 8080 --host 0.0.0.0 --threads 4 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
