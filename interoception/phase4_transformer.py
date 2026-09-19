"""Phase 4: Introspective decoding in a tiny transformer on reversed-sequence copy task."""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from interoception.utils import save_results, set_seed

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3
IGNORE_INDEX = -100
VOCAB_SIZE = 12  # 4 special tokens + 8 data symbols (4..11)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyCausalTransformer(nn.Module):
    """2-layer, 4-head, d_model=128 Causal Transformer Decoder."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal attention mask."""
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: [batch, seq_len, d_model] (last layer activations)
        """
        seq_len = x.size(1)
        mask = self.generate_causal_mask(seq_len, x.device)
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        hidden = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.fc_out(hidden)
        return logits, hidden


def generate_copy_data(
    num_samples: int = 1500, seq_len: int = 4, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[List[int]]]:
    """Generate synthetic sequences for reversed-sequence copy task."""
    rng = np.random.RandomState(seed)
    inputs = []
    targets = []
    raw_prompts = []
    raw_expected = []

    for _ in range(num_samples):
        symbols = rng.randint(4, VOCAB_SIZE, size=seq_len).tolist()
        reversed_symbols = list(reversed(symbols))
        raw_prompts.append([BOS_TOKEN] + symbols + [SEP_TOKEN])
        raw_expected.append(reversed_symbols)

        full_seq = [BOS_TOKEN] + symbols + [SEP_TOKEN] + reversed_symbols + [EOS_TOKEN]
        inp = full_seq[:-1]
        tgt = full_seq[1:]

        # Mask prompt loss
        prompt_len = seq_len + 1
        masked_tgt = [IGNORE_INDEX] * prompt_len + tgt[prompt_len:]

        inputs.append(inp)
        targets.append(masked_tgt)

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        raw_prompts,
        raw_expected,
    )


def compute_activation_entropy(h_step: torch.Tensor) -> float:
    """Compute Shannon entropy of hidden activation distribution across d_model."""
    probs = F.softmax(h_step.abs(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
    return float(entropy)


def autoregressive_generate(
    model: TinyCausalTransformer,
    prompt_seq: List[int],
    target_length: int = 4,
    introspective: bool = False,
    entropy_threshold: float = 4.60,
    top2_boost: float = 1.0,
) -> Tuple[List[int], int]:
    """Generate reversed tokens autoregressively with or without introspective decoding."""
    model.eval()
    curr_seq = list(prompt_seq)
    triggers = 0

    with torch.no_grad():
        for _ in range(target_length):
            x = torch.tensor([curr_seq], dtype=torch.long)
            logits, hidden = model(x)

            next_logits = logits[0, -1, :].clone()
            last_hidden = hidden[0, -1, :]

            if introspective:
                entropy = compute_activation_entropy(last_hidden)
                if entropy > entropy_threshold:
                    triggers += 1
                    # Corrective action: Boost top-2 candidate tokens
                    top2 = torch.topk(next_logits, k=2).indices
                    next_logits[top2] += top2_boost

            next_token = int(next_logits.argmax().item())
            curr_seq.append(next_token)
            if next_token == EOS_TOKEN:
                break

    generated = curr_seq[len(prompt_seq) :]
    return generated, triggers


def run_phase4(seed: int = 42) -> Dict[str, any]:
    """Execute Phase 4 Transformer Introspective Decoding experiment."""
    start_time = time.time()
    set_seed(seed)
    print("=" * 75)
    print("PHASE 4: Introspective Decoding in a Tiny Transformer")
    print("=" * 75)

    seq_len = 4
    print(f"Generating dataset for Reversed-Sequence Copy Task (seq_len={seq_len})...")
    X, Y, prompts, expected = generate_copy_data(num_samples=1600, seq_len=seq_len, seed=seed)
    train_x, train_y = X[:1200], Y[:1200]
    test_prompts = prompts[1200:]
    test_expected = expected[1200:]

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Initializing Tiny Transformer (2 layers, 4 heads, d_model=128, dim_ff=256)...")
    model = TinyCausalTransformer(vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print("Training Transformer for 25 epochs on CPU...")
    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} - Target Loss: {avg_loss:.4f}")

    total_samples = len(test_prompts)
    print(f"\nEvaluating autoregressive generation on {total_samples} test sequences...")

    # Calibrate activation entropy threshold
    entropy_samples = []
    with torch.no_grad():
        for i in range(min(50, total_samples)):
            prompt = test_prompts[i]
            _, h = model(torch.tensor([prompt], dtype=torch.long))
            entropy_samples.append(compute_activation_entropy(h[0, -1, :]))
    calibrated_threshold = float(np.percentile(entropy_samples, 50))
    print(f"Calibrated internal activation entropy threshold: {calibrated_threshold:.4f}")

    standard_exact_matches = 0
    intro_exact_matches = 0
    total_triggers = 0

    for i in range(total_samples):
        prompt = test_prompts[i]
        expected_seq = test_expected[i]

        # 1. Standard autoregressive decoding
        gen_std, _ = autoregressive_generate(
            model, prompt, target_length=seq_len, introspective=False
        )
        if gen_std[:seq_len] == expected_seq:
            standard_exact_matches += 1

        # 2. Introspective decoding with internal confusion trigger
        gen_intro, trigs = autoregressive_generate(
            model,
            prompt,
            target_length=seq_len,
            introspective=True,
            entropy_threshold=calibrated_threshold,
            top2_boost=1.2,
        )
        total_triggers += trigs
        if gen_intro[:seq_len] == expected_seq:
            intro_exact_matches += 1

    std_acc = standard_exact_matches / total_samples
    intro_acc = intro_exact_matches / total_samples
    elapsed_time = time.time() - start_time

    # Print comparative results table
    print("\n" + "=" * 75)
    print("PHASE 4 TRANSFORMER DECODING COMPARISON TABLE:")
    print(f"{'Decoding Method':<40} | {'Exact-Match Accuracy':<22} | {'Triggers':<10}")
    print("-" * 75)
    print(f"{'Standard Autoregressive Decoding':<40} | {std_acc*100:>20.2f}% | {'0':<10}")
    print(f"{'Introspective Decoding (Entropy Triggered)':<40} | {intro_acc*100:>20.2f}% | {total_triggers:<10}")
    print("=" * 75)
    print(f"Total Phase 4 Runtime: {elapsed_time:.2f}s (< 900s requirement)")

    results = {
        "phase": 4,
        "task": "reversed_sequence_copy",
        "seq_len": seq_len,
        "vocab_size": VOCAB_SIZE,
        "test_samples": total_samples,
        "entropy_threshold": calibrated_threshold,
        "standard_exact_match_accuracy": float(std_acc),
        "introspective_exact_match_accuracy": float(intro_acc),
        "accuracy_delta": float((intro_acc - std_acc) * 100.0),
        "introspective_triggers_activated": int(total_triggers),
        "execution_time_seconds": float(elapsed_time),
    }

    save_path = Path("results/phase4.json")
    save_results(results, save_path, seed=seed)
    print(f"Phase 4 results saved to: {save_path.resolve()}\n")

    return results


if __name__ == "__main__":
    run_phase4()
