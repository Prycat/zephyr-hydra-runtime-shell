"""
blackwell/export.py
Registers a GGUF model with Ollama after LoRA training completes.

Pipeline:
  GGUF directory (from lora_steer.py save_pretrained_gguf)
    → find .gguf file
    → write Ollama Modelfile
    → run: ollama create zephyr-steered -f Modelfile
    → print progress lines for the GUI console
"""
import os
import glob
import subprocess

MODEL_NAME = "zephyr-steered"
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELFILE_PATH = os.path.join(_HERE, "Modelfile")


def register_with_ollama(gguf_dir: str) -> bool:
    """
    Given a directory containing a .gguf file, write a Modelfile
    and run `ollama create zephyr-steered`.
    Returns True on success, False on any failure.
    Prints progress lines to stdout for the GUI console.
    """
    if not gguf_dir or not os.path.isdir(gguf_dir):
        print(f"[export] ERROR: gguf_dir does not exist: {gguf_dir}", flush=True)
        return False

    # Find the quantized GGUF file
    matches = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    if not matches:
        print(f"[export] ERROR: No .gguf file found in {gguf_dir}", flush=True)
        return False

    gguf_path = os.path.abspath(matches[0])
    print(f"[export] Found GGUF: {gguf_path}", flush=True)

    # Write Modelfile
    try:
        with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
            f.write(f"FROM {gguf_path}\n")
        print(f"[export] Modelfile written: {MODELFILE_PATH}", flush=True)
    except OSError as e:
        print(f"[export] ERROR: Could not write Modelfile: {e}", flush=True)
        return False

    # Run ollama create
    print(f"[export] Running: ollama create {MODEL_NAME} -f {MODELFILE_PATH}", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH],
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"[export] Model '{MODEL_NAME}' registered in Ollama!", flush=True)
            print(f"[export] Switch to '{MODEL_NAME}' in the model card to use your steered model.",
                  flush=True)
            return True
        else:
            print(f"[export] ERROR: ollama create failed (exit code {result.returncode})",
                  flush=True)
            return False
    except FileNotFoundError:
        print("[export] ERROR: 'ollama' not found in PATH", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[export] ERROR: ollama create timed out after 5 minutes", flush=True)
        return False
    except Exception as e:
        print(f"[export] ERROR: {e}", flush=True)
        return False
