#!/usr/bin/env bash
set -e

echo "=== [1/4] Installing system dependencies ==="
sudo apt update && sudo apt install -y build-essential cmake git curl python3-pip python3-venv

echo "=== [2/4] Setting up Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [3/4] Cloning and building llama.cpp with Arm KleidiAI ==="
if [ ! -d "llama.cpp" ]; then
  git clone --recursive https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build . --config Release -j$(nproc)
cd ../..

echo "=== [4/4] Fetching quantized Llama 3.2 1B Instruct GGUF ==="
mkdir -p models
if [ ! -f "models/llama-3.2-1b-instruct-q4_k_m.gguf" ]; then
  curl -L -o models/llama-3.2-1b-instruct-q4_k_m.gguf \
    https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
fi

echo "=== Setup complete! Execute 'python3 benchmark.py' to test ==="
