"""
Hermes 3 local tool-using agent via vLLM + TurboQuant.
Uses vLLM's OpenAI-compatible API endpoint (localhost:8000).
Run start_server.py before this script.
"""

import json
import math
import datetime
import httpx
from openai import OpenAI

# vLLM exposes an OpenAI-compatible API on port 8000
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",  # vLLM does not require an API key by default
)

MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"

# ─── Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, log, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write the file to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


# ─── Tool implementations ────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    # Safe math eval: expose only math module names
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns["__builtins__"] = {}
    try:
        result = eval(expression, safe_ns)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


TOOL_HANDLERS = {
    "calculate": lambda args: calculate(**args),
    "get_current_time": lambda args: get_current_time(),
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
}


# ─── Agent loop ──────────────────────────────────────────────────────────────

def run_agent(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Run one turn of the agent. Returns (final_reply, updated_history)."""
    history = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        history.append(msg.model_dump(exclude_unset=True))

        # If no tool calls, we have the final answer
        if not msg.tool_calls:
            return msg.content, history

        # Execute each tool call
        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            print(f"  [tool] {fn_name}({fn_args})")

            handler = TOOL_HANDLERS.get(fn_name)
            if handler:
                result = handler(fn_args)
            else:
                result = f"Unknown tool: {fn_name}"

            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })


# ─── Main chat loop ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Hermes, a helpful and capable AI assistant running locally via vLLM with TurboQuant KV cache compression.
You have access to tools: calculate, get_current_time, read_file, write_file.
Use tools whenever they would give a more accurate or useful answer.
Be concise and direct."""


def main():
    print("Hermes 3 Agent (local via vLLM + TurboQuant)")
    print("Type 'exit' or 'quit' to stop, 'clear' to reset history.\n")

    # Verify vLLM server is reachable before entering the chat loop
    try:
        response = httpx.get("http://localhost:8000/health", timeout=3)
        response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        print("ERROR: vLLM server not reachable on port 8000.")
        print("Start it first with:  python start_server.py\n")
        return

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("History cleared.\n")
            continue

        try:
            reply, history = run_agent(user_input, history)
            print(f"\nHermes: {reply}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
