import streamlit as st
import requests
import json
import psutil
import os

SERVER_URL = "http://localhost:8080/completion"

st.set_page_config(
    page_title="Arm KleidiAI SLM Edge Console",
    page_icon="⚡",
    layout="wide"
)

# Sidebar System Telemetry
st.sidebar.title("Arm Hardware Monitor")
cpu_usage = psutil.cpu_percent(interval=None)
ram_info = psutil.virtual_memory()

st.sidebar.metric(label="CPU Utilization", value=f"{cpu_usage}%")
st.sidebar.metric(label="RAM Usage", value=f"{ram_info.percent}% ({round(ram_info.used / (1024**3), 2)} GB)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Engine:** `llama.cpp` + `Arm KleidiAI`")
st.sidebar.markdown("**Instruction Set:** `Neon / SVE2 / SME`")

st.title("⚡ Edge SLM with Arm KleidiAI")
st.caption("Sub-millisecond quantized tensor inference running locally on Arm64 architecture.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your prompt for the local SLM..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        payload = {
            "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "stream": True,
            "n_predict": 256,
            "temperature": 0.6,
            "stop": ["<|eot_id|>", "<|end_of_text|>"]
        }

        try:
            with requests.post(SERVER_URL, json=payload, stream=True, timeout=60) as r:
                for line in r.iter_lines():
                    if line:
                        clean_line = line.decode("utf-8").removeprefix("data: ")
                        try:
                            data = json.loads(clean_line)
                            chunk = data.get("content", "")
                            full_response += chunk
                            response_container.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue
            response_container.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to llama-server. Please verify the inference server is running on port 8080.")
