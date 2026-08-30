import os
import urllib.request
import psutil
import streamlit as st

try:
    from llama_cpp import Llama
except ImportError:
    st.error("Missing dependency! Please ensure llama-cpp-python is listed in requirements.txt.")
    st.stop()

st.set_page_config(
    page_title="Edge SLM Console",
    page_icon="⚡",
    layout="wide"
)

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

# Sidebar Telemetry
st.sidebar.title("Telemetry")
ram = psutil.virtual_memory()
cpu_pct = psutil.cpu_percent(interval=None)

st.sidebar.metric(label="CPU Utilization", value=f"{cpu_pct}%")
st.sidebar.metric(label="RAM Usage", value=f"{ram.percent}% ({round(ram.used / (1024**3), 2)} GB)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Runtime:** Embedded `llama-cpp-python`")

# Chat UI
st.title("⚡ Local Edge SLM Console")
st.caption("Self-contained in-memory inference (zero network dependencies).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter your prompt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
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
            response_box.markdown(full_response + "▌")

        response_box.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
