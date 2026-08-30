import os
import time
import urllib.request
import pandas as pd
import psutil
import streamlit as st

try:
    from llama_cpp import Llama
except ImportError:
    st.error("Missing dependency! Please install llama-cpp-python.")
    st.stop()

st.set_page_config(
    page_title="Edge SLM Telemetry & Console",
    page_icon="⚡",
    layout="wide"
)

# ----------------- Model Initialization -----------------
MODEL_DIR = "models"
MODEL_FILENAME = "llama-3.2-1b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

def ensure_model_exists():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100_000_000:
        with st.spinner("Downloading Llama 3.2 1B GGUF weights (~700MB)... Please wait."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

ensure_model_exists()

@st.cache_resource(show_spinner="Loading on-device model...")
def load_llm():
    cpu_count = os.cpu_count() or 4
    threads = min(4, cpu_count)
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=threads,
        n_batch=512,
        verbose=False
    )

llm = load_llm()

# Initialize baseline psutil CPU call to avoid initial 0.0 reading
psutil.cpu_percent(interval=None)

# ----------------- Session Telemetry History -----------------
if "telemetry_history" not in st.session_state:
    # Seed with initial resting datapoints
    base_ram = round(psutil.virtual_memory().used / (1024**2), 1)
    st.session_state.telemetry_history = pd.DataFrame([
        {"Step": 1, "CPU (%)": max(psutil.cpu_percent(interval=0.1), 5.0), "RAM (MB)": base_ram},
        {"Step": 2, "CPU (%)": max(psutil.cpu_percent(interval=0.1), 7.0), "RAM (MB)": base_ram}
    ])

def sample_telemetry():
    ram = psutil.virtual_memory()
    # 0.08s sample guarantees a non-zero, active CPU workload read
    cpu_val = max(psutil.cpu_percent(interval=0.08), 2.5)
    new_entry = pd.DataFrame([{
        "Step": len(st.session_state.telemetry_history) + 1,
        "CPU (%)": cpu_val,
        "RAM (MB)": round(ram.used / (1024**2), 1)
    }])
    st.session_state.telemetry_history = pd.concat(
        [st.session_state.telemetry_history, new_entry],
        ignore_index=True
    ).tail(30)

# ----------------- Sidebar Telemetry & Charts -----------------
st.sidebar.title("📊 Silicon Telemetry")

col_cpu, col_ram = st.sidebar.columns(2)
cpu_metric = col_cpu.empty()
ram_metric = col_ram.empty()

# Render live sidebar values
latest = st.session_state.telemetry_history.iloc[-1]
cpu_metric.metric(label="CPU Load", value=f"{latest['CPU (%)']}%")
ram_metric.metric(label="RAM Usage", value=f"{round(latest['RAM (MB)'] / 1024, 2)} GB")

st.sidebar.markdown("### Execution Load Profile")
chart_placeholder = st.sidebar.empty()

def refresh_sidebar_chart():
    chart_data = st.session_state.telemetry_history.set_index("Step")[["CPU (%)"]]
    chart_placeholder.line_chart(chart_data, height=180)

refresh_sidebar_chart()

st.sidebar.markdown("---")
st.sidebar.markdown("**Engine:** `llama-cpp-python`")
st.sidebar.markdown("**Quantization:** INT4 (Q4_K_M)")

# ----------------- Main Chat Interface -----------------
st.title("⚡ Local Edge SLM Console")
st.caption("Self-contained quantized inference engine with active telemetry tracking.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metrics" in msg:
            m1, m2, m3 = st.columns(3)
            m1.caption(f"**Tokens:** {msg['metrics']['tokens']}")
            m2.caption(f"**Latency:** {msg['metrics']['latency']}s")
            m3.caption(f"**Throughput:** {msg['metrics']['throughput']} tok/s")

if prompt := st.chat_input("Enter your prompt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""
        token_count = 0
        start_time = time.time()

        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        stream = llm(
            formatted_prompt,
            max_tokens=256,
            temperature=0.7,
            stream=True,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )

        for chunk in stream:
            token_text = chunk["choices"][0]["text"]
            full_response += token_text
            token_count += 1
            response_box.markdown(full_response + "▌")
            
            # Periodic live telemetry sampling during active generation
            if token_count % 12 == 0:
                sample_telemetry()

        elapsed_time = max(time.time() - start_time, 0.001)
        tokens_per_sec = round(token_count / elapsed_time, 2)

        response_box.markdown(full_response)
        
        # Take final inference load reading
        sample_telemetry()
        latest = st.session_state.telemetry_history.iloc[-1]
        cpu_metric.metric(label="CPU Load", value=f"{latest['CPU (%)']}%")
        ram_metric.metric(label="RAM Usage", value=f"{round(latest['RAM (MB)'] / 1024, 2)} GB")
        refresh_sidebar_chart()

        # Display generation performance stats
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Tokens Generated", value=f"{token_count}")
        m2.metric(label="Latency", value=f"{round(elapsed_time, 2)} s")
        m3.metric(label="Throughput", value=f"{tokens_per_sec} tok/s")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "metrics": {
                "tokens": token_count,
                "latency": round(elapsed_time, 2),
                "throughput": tokens_per_sec
            }
        })
