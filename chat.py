"""
chat.py -- Talk to your tiny Claude.

Important honesty note: this model was trained on Shakespeare with a
 CHARACTER tokenizer, so it will CONTINUE text in Shakespeare style -- it is
 not a question-answerer yet. Ask it something and it will reply as if it
 were a lost scene from a play. That's the technology working exactly as
 designed; the same trick at 100,000x the scale (plus human feedback
 training) is what makes Claude.

Fun first experiments:
  - blank prompt (just press Enter): the model starts from nothing
  - "O Romeo, wherefore art  " finish the famous line
  - train it on YOUR text instead (see README) and watch it copy that style

Run it:
    python chat.py
"""
import torch

from model import Config, GPT
from tokenizer import CharTokenizer

CHECKPOINT_PATH = "checkpoints/tinyclaude.pt"
TOKENIZER_PATH = "checkpoints/tokenizer.pt"


def load_model(device):
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def chat_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    lm = Config()

    print()
    print("=" * 64)
    print("  Your tiny Claude is loaded. Type a line and it continues")
    print("  the text, Shakespeare style. Type 'quit' to exit, 'temp 1.5'")
    print("  to set creativity (0.1 rigid ... 1.5 wild).")
    print("=" * 64)
    temperature = 0.8

    while True:
        try:
            prompt = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower().startswith("temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"(creativity set to {temperature})")
            except ValueError:
                print("(usage: temp 0.8)")
            continue

        # Only keep the last (block_size - a bit) characters the model can see.
        ids = tokenizer.encode(prompt)[-lm.block_size + 1 :]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=400, temperature=temperature, top_k=20)
        print("-" * 64)
        print(tokenizer.decode(out[0].tolist()))
        print("-" * 64)

    print("bye!")


if __name__ == "__main__":
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        raise SystemExit("No trained model found. Run:  python data.py  then  python train.py")
    chat_loop()
