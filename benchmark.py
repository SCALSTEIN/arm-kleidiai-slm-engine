import subprocess
import time
import os
import re
import json
import psutil

MODEL_PATH = "models/llama-3.2-1b-instruct-q4_k_m.gguf"
CLI_BIN = "./llama.cpp/build/bin/llama-cli"

PROMPTS = [
    "Explain the architectural advantage of SIMD vectorization in embedded edge processors in three concise bullet points.",
    "Describe the difference between SVE2 and Neon vector execution pipelines.",
    "Draft a Python function using recursion to compute the Fibonacci sequence."
]

def run_single_benchmark(prompt: str, threads: int, n_predict: int = 128):
    if not os.path.exists(CLI_BIN) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Binaries or model weights not found. Run ./setup.sh first.")

    cmd = [
        CLI_BIN,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(n_predict),
        "-t", str(threads),
        "--perf"
    ]

    start_wall = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.perf_counter() - start_wall

    stderr_output = proc.stderr

    # Parse metrics from llama.cpp stderr telemetry
    eval_match = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", stderr_output)
    prompt_match = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens", stderr_output)

    prompt_ms = float(prompt_match.group(1)) if prompt_match else 0.0
    prompt_tokens = int(prompt_match.group(2)) if prompt_match else 0
    eval_tps = float(eval_match.group(4)) if eval_match else 0.0
    eval_tokens = int(eval_match.group(2)) if eval_match else 0

    return {
        "threads": threads,
        "prompt_tokens": prompt_tokens,
        "prompt_eval_ms": prompt_ms,
        "decode_tokens": eval_tokens,
        "decode_tokens_per_sec": eval_tps,
        "total_wall_time_sec": round(wall_time, 2)
    }

def main():
    print("==================================================")
    print("  Arm KleidiAI SLM Performance Benchmark")
    print("==================================================")
    print(f"Detected CPU Cores: {os.cpu_count()}")
    print(f"Target Model: {MODEL_PATH}\n")

    thread_configs = [1, 2, 4]
    if os.cpu_count() and os.cpu_count() >= 8:
        thread_configs.append(8)

    results = []

    for t in thread_configs:
        print(f"[*] Running benchmark with {t} thread(s)...")
        res = run_single_benchmark(PROMPTS[0], threads=t, n_predict=128)
        results.append(res)
        print(f"    -> Prompt Processing: {res['prompt_eval_ms']} ms ({res['prompt_tokens']} tokens)")
        print(f"    -> Generation Throughput: {res['decode_tokens_per_sec']} tokens/sec")
        print(f"    -> Total Wall Time: {res['total_wall_time_sec']} s\n")

    print("=== Final Benchmark Summary ===")
    print(json.dumps(results, indent=2))

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults exported to benchmark_results.json")

if __name__ == "__main__":
    main()
