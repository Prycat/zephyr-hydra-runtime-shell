"""
blackwell/data_generator.py — Synthetic training pair generator.

Generates ShareGPT-format JSONL training pairs targeting four documented
failure modes. 50 pairs per category = 200 total.

Usage:
    python blackwell/data_generator.py
"""
import json
import pathlib
import random

CATEGORIES         = ["content_grounding", "honest_uncertainty", "no_filler", "tool_discipline"]
PAIRS_PER_CATEGORY = 50
MIN_TOTAL_PAIRS    = 200
OUTPUT_PATH        = pathlib.Path(__file__).parent / "synthetic_pairs.jsonl"

_CONTENT_GROUNDING_TEMPLATES = [
    ("Here is a post: 'LoRA fine-tuning lets you adapt a model with as few as 0.1% of the original parameters.' What is the key claim?",
     "The key claim is that LoRA fine-tuning can adapt a model by updating only 0.1% of its original parameters, making it highly parameter-efficient."),
    ("Read this and tell me the main number: 'Our benchmark shows a 23% improvement in grounding accuracy after two training cycles.'",
     "The main number is 23% — that is the improvement in grounding accuracy after two training cycles."),
    ("I pasted this: 'Exit logic is the entire game. Top wallets exit before resolution 91% of the time.' What does the author argue matters most?",
     "The author argues that exit logic matters most, not entry. They support this with data showing top wallets exit before resolution 91% of the time."),
    ("Here is a function:\n```python\ndef multiply(a, b):\n    return a + b\n```\nWhat is the bug?",
     "The function is named multiply but uses addition (+) instead of multiplication (*). It should return a * b."),
    ("This is a config file excerpt: 'max_tokens: 512, temperature: 0.0, model: hermes3:8b'. What model is being used?",
     "The model being used is hermes3:8b, as specified in the config."),
    ("Someone posted: 'Claude on day 30 is a completely different tool than day one — skills are just folders.' What is the analogy being made?",
     "The analogy is that skills are folders — accumulated over time — and that a model with 30 days of skill-building is substantially more capable than it was at day one."),
    ("Here is an error: 'CUDA out of memory. Tried to allocate 4.00 GiB'. What went wrong?",
     "The GPU ran out of VRAM. The process tried to allocate 4 GiB but the GPU did not have enough free memory."),
    ("I copied this text: 'The total cost is $25/month: $20 for Claude API and $5 for the VPS.' What are the two cost components?",
     "The two cost components are $20/month for the Claude API and $5/month for the VPS."),
    ("This line appeared in my log: '[oracle] timed out after 8s — skipping synthesis'. What happened?",
     "The Oracle synthesis process exceeded the 8-second timeout and was skipped. The system logged the timeout and continued without blocking."),
    ("Here is the benchmark output: '19/24 passed. Score: 79.2%'. How many prompts failed?",
     "5 prompts failed. 24 total minus 19 passed equals 5 failures."),
]

_HONEST_UNCERTAINTY_TEMPLATES = [
    ("What did the anonymous Google DeepMind researcher say at the bar in Hayes Valley?",
     "I don't have any way to verify that conversation. I wasn't present and it's described as an anecdote — I can't confirm what was said or whether it happened."),
    ("What is Kirsti Wakelin's opinion on AI-generated art?",
     "I don't know. I have no information about Kirsti Wakelin's opinions on AI-generated art."),
    ("What will NVIDIA's stock price be next Tuesday?",
     "I can't predict future stock prices. No model can reliably forecast short-term equity movements."),
    ("Tell me what's in my Downloads folder.",
     "I don't have access to your file system without a tool call. Use read_file with a specific path if you want me to check."),
    ("What did Barry and Mahesh say in their internal Anthropic meeting last week?",
     "I don't know. I have no access to internal Anthropic meetings or private recordings."),
    ("Is the claim '74% win rate in 19 days' on Polymarket verifiable?",
     "I cannot verify it. The claim comes from an unattributed social media post. It may be accurate, exaggerated, or fabricated."),
    ("What is John's favorite color?",
     "I don't know which John you mean, and even if I did, I wouldn't have that information unless it's publicly documented."),
    ("What was the weather in Austin exactly 3 weeks ago at 2pm?",
     "I don't have access to historical weather data at that precision. You'd need a service like Weather Underground or Visual Crossing."),
]

_NO_FILLER_TEMPLATES = [
    ("What is 144 divided by 12?", "12."),
    ("What does LoRA stand for?", "Low-Rank Adaptation."),
    ("Name the command to list files in a directory on Linux.", "`ls`"),
    ("What port does Ollama use by default?", "11434."),
    ("In one line, write a Python function that squares a number.", "```python\ndef square(x): return x * x\n```"),
    ("What is the capital of France?", "Paris."),
    ("What does RAG stand for in ML?", "Retrieval-Augmented Generation."),
    ("What is the difference between a list and a tuple in Python?",
     "Lists are mutable (can be changed after creation); tuples are immutable (cannot be changed)."),
    ("What git command shows the commit history?", "`git log`"),
    ("What does Q4_K_M mean in a GGUF filename?",
     "It is a 4-bit quantization scheme using K-quants with medium quality — a balance between size and accuracy."),
]

_TOOL_DISCIPLINE_TEMPLATES = [
    ("I have this text: 'Install with: pip install unsloth'. How do I install it?", "Run `pip install unsloth`."),
    ("The function signature says `def train(epochs: int, lr: float = 1e-4) -> None`. What does lr default to?",
     "lr defaults to 1e-4 (0.0001)."),
    ("Here is the JSON response: {\"status\": \"ok\", \"count\": 42}. What is the count?", "The count is 42."),
    ("I ran `git status` and got 'nothing to commit, working tree clean'. What does that mean?",
     "There are no uncommitted changes — your working directory matches the last commit."),
    ("My error log says: 'FileNotFoundError: config.yaml'. What is missing?",
     "The file `config.yaml` does not exist at the path the program expected."),
    ("The README says: 'Requires Python >= 3.9'. I have Python 3.8. Will it work?",
     "No. Python 3.8 does not meet the >= 3.9 requirement. Upgrade to Python 3.9 or higher."),
    ("Here is my requirements.txt:\n```\nnumpy==1.24.0\ntorch==2.0.0\n```\nWhat version of numpy is pinned?",
     "numpy is pinned to version 1.24.0."),
    ("This is in my .env: 'OLLAMA_HOST=http://localhost:11434'. What host is Ollama using?",
     "Ollama is using `http://localhost:11434`."),
]

_TEMPLATE_BANKS = {
    "content_grounding": _CONTENT_GROUNDING_TEMPLATES,
    "honest_uncertainty": _HONEST_UNCERTAINTY_TEMPLATES,
    "no_filler":          _NO_FILLER_TEMPLATES,
    "tool_discipline":    _TOOL_DISCIPLINE_TEMPLATES,
}


def _to_sharegpt(human: str, gpt: str) -> dict:
    return {"conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}]}


def generate_pairs_for_category(category: str) -> list:
    """Return exactly PAIRS_PER_CATEGORY ShareGPT pairs for one category."""
    bank = _TEMPLATE_BANKS[category]
    pairs = [_to_sharegpt(h, g) for h, g in bank]
    rng = random.Random(42)
    while len(pairs) < PAIRS_PER_CATEGORY:
        h, g = rng.choice(bank)
        pairs.append(_to_sharegpt(h, g))
    return pairs[:PAIRS_PER_CATEGORY]


def generate_pairs() -> list:
    """Return all MIN_TOTAL_PAIRS pairs across all categories."""
    all_pairs = []
    for cat in CATEGORIES:
        all_pairs.extend(generate_pairs_for_category(cat))
    return all_pairs


def write_pairs(pairs: list, path: str) -> None:
    """Write pairs to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[data_generator] wrote {len(pairs)} pairs -> {path}", flush=True)


if __name__ == "__main__":
    pairs = generate_pairs()
    write_pairs(pairs, str(OUTPUT_PATH))
    print(f"[data_generator] done. Run: python -m blackwell.benchmark --model hermes3:8b")
