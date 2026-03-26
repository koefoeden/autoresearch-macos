"""
Minimal chat/inference script for a locally trained checkpoint.

Usage:
    uv run chat.py
    uv run chat.py --prompt "Hello"
"""

import argparse
import os
import pickle
import sys

import torch

from model import GPT, GPTConfig


CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
CHECKPOINT_PATH = os.path.join(CACHE_DIR, "checkpoints", "latest.pt")
BOS_TOKEN = "<|reserved_0|>"


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(tokenizer_dir=TOKENIZER_DIR):
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)
    return enc, enc.encode_single_token(BOS_TOKEN)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_k, eos_text=None):
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.sequence_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

        if eos_text is not None:
            text = decode_tokens(idx[0].tolist())
            if eos_text in text:
                break

    return idx[0].tolist()


def decode_tokens(token_ids):
    return TOKENIZER.decode(token_ids)


def build_prompt_ids(text):
    token_ids = TOKENIZER.encode_ordinary(text)
    token_ids = token_ids[-(CONFIG.sequence_len - 1):]
    return [BOS_TOKEN_ID] + token_ids


def run_prompt(prompt, max_new_tokens, temperature, top_k):
    prompt_ids = build_prompt_ids(prompt)
    output_ids = generate(
        MODEL,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    new_text = decode_tokens(output_ids[len(prompt_ids):])
    return new_text


def interactive_chat(max_new_tokens, temperature, top_k):
    print("Loaded checkpoint. Type 'exit' or 'quit' to stop.")
    history = ""
    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            return

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return

        history += f"User: {user_text}\nAssistant:"
        response = run_prompt(history, max_new_tokens, temperature, top_k)
        stop_marker = "\nUser:"
        if stop_marker in response:
            response = response.split(stop_marker, 1)[0]
        response = response.strip()
        print(f"Assistant: {response}")
        history += f" {response}\n"


def main():
    global DEVICE, TOKENIZER, BOS_TOKEN_ID, MODEL, CONFIG, CHECKPOINT

    parser = argparse.ArgumentParser(description="Chat with the latest trained autoresearch checkpoint")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Checkpoint path")
    parser.add_argument("--prompt", default=None, help="One-shot prompt instead of interactive chat")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Maximum tokens to sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff (0 disables)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}")
        print("Run `uv run train.py` once after this change to create a chat checkpoint.")
        sys.exit(1)

    DEVICE = detect_device()
    TOKENIZER, BOS_TOKEN_ID = load_tokenizer()
    MODEL, CONFIG, CHECKPOINT = load_model(args.checkpoint, DEVICE)

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"val_bpb={CHECKPOINT['val_bpb']:.6f} | steps={CHECKPOINT['num_steps']}")

    if args.prompt is not None:
        print(run_prompt(args.prompt, args.max_new_tokens, args.temperature, args.top_k))
    else:
        interactive_chat(args.max_new_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
