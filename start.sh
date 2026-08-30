#!/usr/bin/env bash
set -e

MODEL_PATH="models/llama-3.2-1b-instruct-q4_k_m.gguf"
SERVER_BIN="./llama.cpp/build/bin/llama-server"
PORT=8080
HOST="127.0.0.1"
HEALTH_URL="http://${HOST}:${PORT}/health"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Check prerequisites
if [ ! -f "$SERVER_BIN" ]; then
    echo "[!] Error: Binary $SERVER_BIN not found. Please build llama.cpp first."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[!] Error: Model $MODEL_PATH not found. Please download model weights first."
    exit 1
fi

# Cleanup handler: kill background llama-server on script exit/Ctrl+C
cleanup() {
    echo ""
    echo "[*] Shutting down services..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null
        echo "[*] llama-server (PID $SERVER_PID) terminated."
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Activate Python virtual environment if available
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Determine optimal thread count (default to min(4, CPU cores))
CORES=$(nproc 2>/dev/null || echo 4)
THREADS=$(( CORES > 4 ? 4 : CORES ))

echo "=================================================="
echo " Starting Edge SLM Services (Arm KleidiAI)"
echo "=================================================="
echo "[*] Target Model: $MODEL_PATH"
echo "[*] Threads: $THREADS | Host: $HOST | Port: $PORT"

# Launch llama-server in the background
$SERVER_BIN \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    -c 2048 \
    --threads "$THREADS" > llama_server.log 2>&1 &

SERVER_PID=$!
echo "[*] llama-server started in background (PID: $SERVER_PID)"
echo "[*] Waiting for inference endpoint to become ready..."

# Polling loop for server health
COUNT=0
READY=false

while [ $COUNT -lt $MAX_RETRIES ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[!] llama-server exited unexpectedly. Check llama_server.log for details:"
        tail -n 20 llama_server.log
        exit 1
    fi

    # Check HTTP health status
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        READY=true
        break
    fi

    COUNT=$((COUNT + 1))
    sleep $RETRY_INTERVAL
done

if [ "$READY" = false ]; then
    echo "[!] Timeout waiting for llama-server on port $PORT. Check llama_server.log:"
    tail -n 20 llama_server.log
    exit 1
fi

echo "[✓] Backend is live at http://${HOST}:${PORT}"
echo "[*] Launching Streamlit UI on http://localhost:8501..."
echo "=================================================="

# Run Streamlit in the foreground
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
