"""
Main Training & Evaluation Script
==================================
Trains both Seq2Seq models (with and without attention),
generates comparison metrics, plots, and attention heatmaps.

Usage: python train_and_evaluate.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from models import (EncoderNoAttn, DecoderNoAttn, Seq2SeqNoAttention,
                    EncoderWithAttn, DecoderWithAttn, Seq2SeqWithAttention)
from dataset import get_data, PAD_TOKEN

# ── Hyperparameters ──
EMB_DIM = 128
HIDDEN_DIM = 256
ENC_HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
N_EPOCHS = 60
LR = 0.001
MAX_LEN = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print("=" * 60)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, clip, has_attention):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for src, trg in loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        if has_attention:
            output, _ = model(src, trg)
        else:
            output = model(src, trg)
        # output: (batch, trg_len, vocab)
        output_flat = output[:, 1:].contiguous().view(-1, output.shape[-1])
        trg_flat = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # Accuracy (non-pad tokens)
        preds = output_flat.argmax(1)
        mask = trg_flat != 0  # PAD=0
        correct += (preds[mask] == trg_flat[mask]).sum().item()
        total += mask.sum().item()

    return epoch_loss / len(loader), correct / total if total > 0 else 0


def evaluate(model, loader, criterion, has_attention):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            if has_attention:
                output, _ = model(src, trg, teacher_forcing_ratio=0)
            else:
                output = model(src, trg, teacher_forcing_ratio=0)
            output_flat = output[:, 1:].contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()
            preds = output_flat.argmax(1)
            mask = trg_flat != 0
            correct += (preds[mask] == trg_flat[mask]).sum().item()
            total += mask.sum().item()

    return epoch_loss / len(loader), correct / total if total > 0 else 0


def compute_bleu_simple(references, hypotheses):
    """Simple BLEU-like score (unigram precision)."""
    if not references:
        return 0.0
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if len(hyp_tokens) == 0:
            scores.append(0.0)
            continue
        matches = sum(1 for t in hyp_tokens if t in ref_tokens)
        precision = matches / len(hyp_tokens)
        brevity = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
        scores.append(precision * brevity)
    return sum(scores) / len(scores) * 100


def translate_sentence(model, src_tensor, src_vocab, trg_vocab, has_attention, max_len=20):
    """Translate a single sentence."""
    model.eval()
    with torch.no_grad():
        src_tensor = src_tensor.unsqueeze(0).to(DEVICE)
        if has_attention:
            enc_outputs, hidden, cell = model.encoder(src_tensor)
        else:
            hidden, cell = model.encoder(src_tensor)

        input_tok = torch.tensor([trg_vocab.word2idx["<sos>"]]).to(DEVICE)
        output_tokens = []
        attn_weights_all = []

        for _ in range(max_len):
            if has_attention:
                output, hidden, cell, attn_w = model.decoder(input_tok, hidden, cell, enc_outputs)
                attn_weights_all.append(attn_w.cpu().numpy())
            else:
                output, hidden, cell = model.decoder(input_tok, hidden, cell)

            top1 = output.argmax(1).item()
            if top1 == trg_vocab.word2idx["<eos>"]:
                break
            output_tokens.append(trg_vocab.idx2word.get(top1, "<unk>"))
            input_tok = torch.tensor([top1]).to(DEVICE)

    attn_matrix = np.concatenate(attn_weights_all, axis=0) if attn_weights_all else None
    return " ".join(output_tokens), attn_matrix


def plot_attention(attention, src_words, trg_words, filename):
    """Plot attention heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    src_len = len(src_words)
    trg_len = len(trg_words)
    attn = attention[:trg_len, :src_len]

    cax = ax.matshow(attn, cmap='YlOrRd', aspect='auto')
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(src_len))
    ax.set_yticks(range(trg_len))
    ax.set_xticklabels(src_words, rotation=45, ha='left', fontsize=11)
    ax.set_yticklabels(trg_words, fontsize=11)
    ax.set_xlabel('Source (English)', fontsize=13)
    ax.set_ylabel('Target (French)', fontsize=13)
    ax.set_title('Bahdanau Attention Weights', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, filename):
    """Plot loss and accuracy curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['no_attn']['train_loss'], label='Without Attention (Train)', color='#e74c3c', linewidth=2)
    axes[0].plot(history['no_attn']['val_loss'], label='Without Attention (Val)', color='#e74c3c', linestyle='--', linewidth=2)
    axes[0].plot(history['attn']['train_loss'], label='With Attention (Train)', color='#2ecc71', linewidth=2)
    axes[0].plot(history['attn']['val_loss'], label='With Attention (Val)', color='#2ecc71', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['no_attn']['train_acc'], label='Without Attention (Train)', color='#e74c3c', linewidth=2)
    axes[1].plot(history['no_attn']['val_acc'], label='Without Attention (Val)', color='#e74c3c', linestyle='--', linewidth=2)
    axes[1].plot(history['attn']['train_acc'], label='With Attention (Train)', color='#2ecc71', linewidth=2)
    axes[1].plot(history['attn']['val_acc'], label='With Attention (Val)', color='#2ecc71', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Seq2Seq Model Comparison: With vs Without Attention', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("Loading data...")
    train_loader, test_loader, src_vocab, trg_vocab, train_pairs, test_pairs = get_data(BATCH_SIZE, MAX_LEN)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    print("=" * 60)

    pad_idx = src_vocab.word2idx[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    history = {
        'no_attn': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
        'attn': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    }

    # ══════════════════════════════════════════════════════════
    # TRAIN MODEL 1: Without Attention
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1: Seq2Seq WITHOUT Attention")
    print("=" * 60)

    enc_no = EncoderNoAttn(len(src_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    dec_no = DecoderNoAttn(len(trg_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    model_no_attn = Seq2SeqNoAttention(enc_no, dec_no, DEVICE).to(DEVICE)
    optimizer_no = optim.Adam(model_no_attn.parameters(), lr=LR)

    print(f"Parameters: {count_parameters(model_no_attn):,}")
    start_time_no = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model_no_attn, train_loader, optimizer_no, criterion, 1.0, False)
        v_loss, v_acc = evaluate(model_no_attn, test_loader, criterion, False)
        history['no_attn']['train_loss'].append(t_loss)
        history['no_attn']['val_loss'].append(v_loss)
        history['no_attn']['train_acc'].append(t_acc)
        history['no_attn']['val_acc'].append(v_acc)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    time_no_attn = time.time() - start_time_no
    print(f"  Training time: {time_no_attn:.2f}s")

    # ══════════════════════════════════════════════════════════
    # TRAIN MODEL 2: With Attention
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2: Seq2Seq WITH Bahdanau Attention")
    print("=" * 60)

    enc_attn = EncoderWithAttn(len(src_vocab), EMB_DIM, ENC_HIDDEN, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    dec_attn = DecoderWithAttn(len(trg_vocab), EMB_DIM, ENC_HIDDEN, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    model_attn = Seq2SeqWithAttention(enc_attn, dec_attn, DEVICE).to(DEVICE)
    optimizer_attn = optim.Adam(model_attn.parameters(), lr=LR)

    print(f"Parameters: {count_parameters(model_attn):,}")
    start_time_attn = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model_attn, train_loader, optimizer_attn, criterion, 1.0, True)
        v_loss, v_acc = evaluate(model_attn, test_loader, criterion, True)
        history['attn']['train_loss'].append(t_loss)
        history['attn']['val_loss'].append(v_loss)
        history['attn']['train_acc'].append(t_acc)
        history['attn']['val_acc'].append(v_acc)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    time_attn = time.time() - start_time_attn
    print(f"  Training time: {time_attn:.2f}s")

    # ══════════════════════════════════════════════════════════
    # EVALUATION & COMPARISON
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVALUATION & COMPARISON")
    print("=" * 60)

    # Plot training curves
    plot_training_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    print("Saved: results/training_curves.png")

    # Translation samples
    refs_no, hyps_no = [], []
    refs_attn, hyps_attn = [], []
    sample_translations = []

    for i, (src_sent, trg_sent) in enumerate(test_pairs[:15]):
        src_enc = src_vocab.encode(src_sent, MAX_LEN)
        src_tensor = torch.tensor(src_enc)

        pred_no, _ = translate_sentence(model_no_attn, src_tensor, src_vocab, trg_vocab, False)
        pred_attn, attn_matrix = translate_sentence(model_attn, src_tensor, src_vocab, trg_vocab, True)

        refs_no.append(trg_sent)
        hyps_no.append(pred_no)
        refs_attn.append(trg_sent)
        hyps_attn.append(pred_attn)

        sample_translations.append({
            'source': src_sent,
            'reference': trg_sent,
            'no_attention': pred_no,
            'with_attention': pred_attn
        })

        # Plot attention for first 3 test sentences
        if i < 3 and attn_matrix is not None:
            src_words = ["<sos>"] + src_sent.split() + ["<eos>"]
            trg_words = pred_attn.split()
            if len(trg_words) > 0:
                plot_attention(attn_matrix, src_words[:attn_matrix.shape[1]],
                              trg_words, os.path.join(RESULTS_DIR, f"attention_heatmap_{i+1}.png"))
                print(f"Saved: results/attention_heatmap_{i+1}.png")

    bleu_no = compute_bleu_simple(refs_no, hyps_no)
    bleu_attn = compute_bleu_simple(refs_attn, hyps_attn)

    final_no_loss = history['no_attn']['val_loss'][-1]
    final_no_acc = history['no_attn']['val_acc'][-1]
    final_attn_loss = history['attn']['val_loss'][-1]
    final_attn_acc = history['attn']['val_acc'][-1]

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Metric':<25} {'Without Attention':>20} {'With Attention':>20}")
    print("-" * 65)
    print(f"{'Final Val Loss':<25} {final_no_loss:>20.4f} {final_attn_loss:>20.4f}")
    print(f"{'Final Val Accuracy':<25} {final_no_acc:>20.4f} {final_attn_acc:>20.4f}")
    print(f"{'BLEU Score':<25} {bleu_no:>20.2f} {bleu_attn:>20.2f}")
    print(f"{'Training Time (s)':<25} {time_no_attn:>20.2f} {time_attn:>20.2f}")
    print(f"{'Parameters':<25} {count_parameters(model_no_attn):>20,} {count_parameters(model_attn):>20,}")

    # Sample translations
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)
    for s in sample_translations[:8]:
        print(f"  SRC: {s['source']}")
        print(f"  REF: {s['reference']}")
        print(f"  NO ATTN: {s['no_attention']}")
        print(f"  ATTN:    {s['with_attention']}")
        print()

    # Save results to JSON
    results = {
        'metrics': {
            'no_attention': {
                'val_loss': final_no_loss, 'val_accuracy': final_no_acc,
                'bleu': bleu_no, 'training_time': time_no_attn,
                'parameters': count_parameters(model_no_attn)
            },
            'with_attention': {
                'val_loss': final_attn_loss, 'val_accuracy': final_attn_acc,
                'bleu': bleu_attn, 'training_time': time_attn,
                'parameters': count_parameters(model_attn)
            }
        },
        'sample_translations': sample_translations,
        'history': {
            'no_attn': {k: [float(v) for v in vals] for k, vals in history['no_attn'].items()},
            'attn': {k: [float(v) for v in vals] for k, vals in history['attn'].items()}
        }
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/results.json")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ['Validation Loss', 'Validation Accuracy', 'BLEU Score']
    no_vals = [final_no_loss, final_no_acc * 100, bleu_no]
    attn_vals = [final_attn_loss, final_attn_acc * 100, bleu_attn]
    colors = ['#e74c3c', '#2ecc71']

    for i, (name, nv, av) in enumerate(zip(metrics_names, no_vals, attn_vals)):
        bars = axes[i].bar(['Without\nAttention', 'With\nAttention'], [nv, av], color=colors, width=0.5, edgecolor='white')
        axes[i].set_title(name, fontsize=13, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, [nv, av]):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Model Performance Comparison', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_chart.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/comparison_chart.png")

    print("\n" + "=" * 60)
    print("ALL DONE! Check the 'results/' folder for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
