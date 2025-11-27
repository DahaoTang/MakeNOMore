# %% [markdown]
# # Neural Operator vs. WaveNet (Makemore Part 5 Remastered)
#
# **Goal:** Compare Andrej Karpathy's "WaveNet" (Hierarchical Dilated CNN) against a Fourier Neural Operator (FNO) on the names dataset.
#
# **Hypothesis:** The FNO should achieve similar or better performance with a "flatter" architecture because the Fourier Transform has a global receptive field immediately.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import requests
import random
import numpy as np

# --- Configuration ---
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

# Increased context length to test long-range dependencies
BLOCK_SIZE = 32
BATCH_SIZE = 32
MAX_STEPS = 200000
EMBED_DIM = 64
HIDDEN_DIM = 64
EVAL_INTERVAL = 500  # How often to evaluate validation loss

# %% [markdown]
# ## 1. Data Preparation

# %%
if not os.path.exists('names.txt'):
    print("Downloading names.txt...")
    r = requests.get(
        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt')
    with open('names.txt', 'wb') as f:
        f.write(r.content)

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

print(f"Vocab size: {vocab_size}")
print(f"Context length: {BLOCK_SIZE}")


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.shuffle(words)
n1 = int(0.9*len(words))
n2 = int(0.95*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xval, Yval = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(f"Train: {Xtr.shape}, Val: {Xval.shape}")

# %% [markdown]
# ## 2. The Baseline: WaveNet (Fixed)

# %%


class FlattenConsecutive(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        B, T, C = x.shape
        # FIX: Use .reshape() instead of .view() to handle non-contiguous memory
        # created by the permute() in TemporalBatchNorm1d
        x = x.reshape(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


class TemporalBatchNorm1d(nn.Module):
    """
    Standard BatchNorm1d expects (N, C, L), but Linear outputs (N, L, C).
    This wrapper permutes dimensions automatically.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.ndim == 2:
            return self.bn(x)
        # Permute: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        # Permute back: (B, C, T) -> (B, T, C)
        return x.permute(0, 2, 1)


class WaveNetBaseline(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

        self.net = nn.Sequential(
            # 32 -> 16
            FlattenConsecutive(2),
            nn.Linear(n_embd * 2, n_hidden, bias=False),
            TemporalBatchNorm1d(n_hidden),
            nn.Tanh(),
            # 16 -> 8
            FlattenConsecutive(2),
            nn.Linear(n_hidden * 2, n_hidden, bias=False),
            TemporalBatchNorm1d(n_hidden),
            nn.Tanh(),
            # 8 -> 4
            FlattenConsecutive(2),
            nn.Linear(n_hidden * 2, n_hidden, bias=False),
            TemporalBatchNorm1d(n_hidden),
            nn.Tanh(),
            # 4 -> 2
            FlattenConsecutive(2),
            nn.Linear(n_hidden * 2, n_hidden, bias=False),
            TemporalBatchNorm1d(n_hidden),
            nn.Tanh(),
            # 2 -> 1
            FlattenConsecutive(2),
            nn.Linear(n_hidden * 2, n_hidden, bias=False),
            TemporalBatchNorm1d(n_hidden),
            nn.Tanh(),
            # Output
            nn.Linear(n_hidden, vocab_size)
        )

        with torch.no_grad():
            self.net[-1].weight *= 0.1

    def forward(self, x):
        x = self.embedding(x)
        logits = self.net(x)
        return logits

# %% [markdown]
# ## 3. The Challenger: FNO Model

# %%


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        # 1. FFT
        x_ft = torch.fft.rfft(x)

        # 2. Multiply relevant modes
        actual_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :actual_modes] = torch.einsum("bix,iox->box",
                                                   x_ft[:, :, :actual_modes],
                                                   self.weights[:, :, :actual_modes])

        # 3. IFFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, modes=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # FNO Layer
        self.fno = SpectralConv1d(n_embd, n_embd, modes)

        # Mixing Layer (1x1 Conv)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_embd, 1),
            nn.GELU(),
            nn.Conv1d(n_embd, n_embd, 1)
        )

        self.head = nn.Linear(n_embd * block_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embedding(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb

        # FNO expects (Batch, Channels, Time)
        x = x.permute(0, 2, 1)

        x = x + self.fno(x)  # Global mixing
        x = x + self.mlp(x)  # Local mixing

        x = x.flatten(1)
        logits = self.head(x)
        return logits

# %% [markdown]
# ## 4. Training Engine

# %%


def train_and_evaluate(model_name, model, steps=2000):
    print(f"\n--- Training {model_name} ---")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_history = []
    val_history = []
    best_val_loss = float('inf')

    model.train()
    for i in range(steps):
        # Batch
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
        Xb, Yb = Xtr[ix].to(device), Ytr[ix].to(device)

        # Forward
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_history.append(loss.item())

        # Evaluation
        if (i+1) % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                # Val Loss
                ix_val = torch.randint(0, Xval.shape[0], (500,))
                Xv, Yv = Xval[ix_val].to(device), Yval[ix_val].to(device)
                val_loss = F.cross_entropy(model(Xv), Yv).item()
                val_history.append((i, val_loss))

                print(
                    f"{i+1}/{steps} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{model_name}_best.pth")
            model.train()

    return train_history, val_history

# --- Run Experiments ---


wavenet = WaveNetBaseline(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
train_hist_wn, val_hist_wn = train_and_evaluate(
    "WaveNet", wavenet, steps=MAX_STEPS)

fno = FNOModel(vocab_size, EMBED_DIM, BLOCK_SIZE, modes=16).to(device)
train_hist_fno, val_hist_fno = train_and_evaluate("FNO", fno, steps=MAX_STEPS)

# %% [markdown]
# ## 5. Visualization & Results

# %%


def smooth(scalars, weight=0.95):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Plot Training Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(smooth(train_hist_wn), label='WaveNet Train', alpha=0.7)
plt.plot(smooth(train_hist_fno), label='FNO Train', alpha=0.7)
plt.title("Training Loss (Smoothed)")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Validation Loss
plt.subplot(1, 2, 2)
# Unzip val history
wn_x, wn_y = zip(*val_hist_wn)
fno_x, fno_y = zip(*val_hist_fno)

plt.plot(wn_x, wn_y, 'o-', label='WaveNet Val', markersize=4)
plt.plot(fno_x, fno_y, 'o-', label='FNO Val', markersize=4)
plt.title("Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_comparison.png")
print("\n[Info] Diagram saved as 'loss_comparison.png'")
plt.show()

# %% [markdown]
# ## 6. Final Loss Comparison Table

# %%
# Get final validation losses
final_val_wn = val_hist_wn[-1][1]
final_val_fno = val_hist_fno[-1][1]

print("\n" + "="*40)
print("FINAL RESULTS COMPARISON")
print("="*40)
print(f"{'Model':<15} | {'Final Val Loss':<15}")
print("-" * 32)
print(f"{'WaveNet':<15} | {final_val_wn:.5f}")
print(f"{'FNO':<15} | {final_val_fno:.5f}")
print("-" * 32)

winner = "WaveNet" if final_val_wn < final_val_fno else "FNO"
print(f"Winner: {winner}")
print("="*40)

# %% [markdown]
# ## 7. Generation Test

# %%


def generate_names(model, model_path, num=5):
    print(f"\n--- Generating with {model_path} ---")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("Warning: Model file not found, using current weights.")

    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(42)

    for _ in range(num):
        out = []
        context = [0] * BLOCK_SIZE
        while True:
            x = torch.tensor([context], device=device)
            with torch.no_grad():
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out[:-1]))


generate_names(wavenet, "WaveNet_best.pth")
generate_names(fno, "FNO_best.pth")
