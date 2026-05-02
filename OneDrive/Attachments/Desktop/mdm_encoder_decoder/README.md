# Encoder–Decoder Models with and without Attention

A comprehensive implementation comparing Sequence-to-Sequence (Seq2Seq) encoder-decoder architectures **with** and **without** attention mechanisms for machine translation. This project demonstrates how attention mechanisms enable neural networks to dynamically focus on different parts of input sequences, improving translation quality.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Amirfurquani/202301070165_AMIR_Encoder-Decoder-Models-with-and-without-Attention.git
cd encoder-decoder-models
pip install -r requirements.txt

# Run the complete pipeline
python train_and_evaluate.py
```

This will train both models, evaluate them, and generate comparison visualizations in the `/results` directory.

## Overview

This project is part of the **Modelling and Deep Machine Learning (MDM) Lab Report** and implements two Seq2Seq architectures for English-French translation:

| Model | Architecture | Context | Best For |
|-------|--------------|---------|----------|
| **Baseline** | Unidirectional LSTM → Fixed Context | Final hidden state only | Short sequences, simplicity |
| **Attention** | BiLSTM → Dynamic Attention Context | Learned alignment weights | Longer sequences, better quality |

## Project Structure

```
.
├── dataset.py                 # Data utilities and English-French parallel corpus
├── models.py                  # Encoder-Decoder model architectures
├── train_and_evaluate.py      # Training and evaluation script
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.9.0+
- NumPy, Matplotlib

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Amirfurquani/202301070165_AMIR_Encoder-Decoder-Models-with-and-without-Attention.git
cd encoder-decoder-models

# Install required packages
pip install -r requirements.txt
```

### Requirements
See `requirements.txt` for complete dependency list:
- `torch>=1.9.0` - Deep learning framework
- `numpy>=1.19.0` - Numerical computing
- `matplotlib>=3.3.0` - Visualization
- `torchvision>=0.10.0` - Computer vision utilities

## Usage

### Running the Complete Pipeline
```bash
python train_and_evaluate.py
```

**What happens:**
1. ✅ Loads English-French parallel corpus (100+ sentence pairs)
2. ✅ Builds source and target vocabularies
3. ✅ Initializes both baseline and attention models
4. ✅ Trains both models for 60 epochs
5. ✅ Evaluates on test set with multiple metrics
6. ✅ Generates comparison visualizations
7. ✅ Saves results to `/results` directory

**Output Files Generated:**
- `training_metrics.json` - Loss and accuracy logs for both models
- `loss_curves.png` - Training/validation loss comparison
- `model_comparison.png` - Side-by-side performance analysis
- `attention_heatmaps.png` - Attention weight visualizations

## Architecture Details

### Model 1: Baseline (Without Attention)

```
Input → Encoder (LSTM) → Final Hidden State → Decoder (LSTM) → Output
```

**Components:**
- `EncoderNoAttn`: Single-layer unidirectional LSTM encoder
- `DecoderNoAttn`: LSTM decoder using only the final encoder hidden state as context
- `Seq2SeqNoAttention`: Complete end-to-end baseline model

**Characteristics:**
- Simpler architecture with fewer parameters
- Fixed bottleneck: must compress all input information into single context vector
- Faster training and inference
- Struggles with long sequences (loses early context)

### Model 2: Attention-Based (Bahdanau Attention)

```
Input → BiEncoder (LSTM) → Attention Mechanism → Decoder (LSTM) → Output
                ↑___________________|
```

**Components:**
- `EncoderWithAttn`: Bidirectional LSTM encoder producing all hidden states
- `AttentionLayer`: Bahdanau additive attention mechanism
- `DecoderWithAttn`: LSTM decoder with attention context at each time step
- `Seq2SeqWithAttention`: Complete attention-based model

**Characteristics:**
- Dynamic context: learns which encoder outputs to focus on at each decoding step
- Bidirectional encoding: captures context from both directions
- Interpretable: attention weights show alignment between input and output
- Better for longer sequences and handles the bottleneck problem

### Attention Mechanism (Bahdanau)

The attention mechanism computes a context vector dynamically at each decoder step:

**Score Function:**
$$\text{score}(h_t, \bar{s}_i) = v^T \tanh(W_1 h_t + W_2 \bar{s}_i)$$

**Attention Weights (Softmax):**
$$\alpha_{t,i} = \frac{\exp(\text{score}(h_t, \bar{s}_i))}{\sum_j \exp(\text{score}(h_t, \bar{s}_j))}$$

**Context Vector:**
$$c_t = \sum_i \alpha_{t,i} \bar{s}_i$$

Where:
- $h_t$ = decoder hidden state at time $t$
- $\bar{s}_i$ = encoder output at position $i$
- $c_t$ = context vector (weighted sum of encoder outputs)
- $\alpha_{t,i}$ = attention weight showing importance of input position $i$
- $v, W_1, W_2$ = learnable parameters

## Features by Module

### `dataset.py` - Data Handling
- **English-French Corpus**: 100+ carefully curated parallel sentence pairs
- **Vocabulary Management**: Automatic vocabulary construction with special tokens
- **Tokenization**: Word-level tokenization with PAD, SOS (Start-of-Sequence), EOS (End-of-Sequence) tokens
- **Data Preprocessing**: Sequence padding and batching
- **PyTorch Integration**: Native DataLoader compatibility for efficient batch processing

### `models.py` - Model Architectures
- Complete implementation of both encoder-decoder models
- Modular design: separate encoder, decoder, and attention components
- Weight initialization for stable training
- Flexible hyperparameter configuration

### `train_and_evaluate.py` - Training & Evaluation Pipeline
- **Training Loop**: Complete training with teacher forcing strategy
- **Evaluation Metrics**: 
  - Perplexity and loss computation
  - Token-level accuracy (excluding padding tokens)
  - Per-sequence accuracy
- **Visualization**: 
  - Training/validation loss curves
  - Model performance comparison charts
  - Attention weight heatmaps for interpretability
- **Results Storage**: All metrics and visualizations saved for analysis

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMB_DIM` | 128 | Embedding vector dimension |
| `HIDDEN_DIM` | 256 | Hidden state size for baseline model |
| `ENC_HIDDEN` | 128 | Encoder hidden dimension for attention model |
| `N_LAYERS` | 2 | Number of LSTM stacked layers |
| `DROPOUT` | 0.3 | Dropout rate for regularization |
| `BATCH_SIZE` | 32 | Training batch size |
| `N_EPOCHS` | 60 | Total training epochs |
| `LR` | 0.001 | Adam optimizer learning rate |
| `MAX_LEN` | 20 | Maximum sequence length (padding/truncation) |

### Training Details

**Optimization:**
- Optimizer: Adam (adaptive learning rate)
- Loss Function: CrossEntropyLoss
- Gradient Clipping: Norm clipped at 1.0 (prevent exploding gradients)

**Teacher Forcing:**
- Training: 50% probability of using ground truth tokens
- Evaluation: 0% (fully autoregressive decoding)
- Purpose: Balances stability and exposure to model errors

## Results & Comparison

### Performance Comparison

| Metric | Baseline | With Attention |
|--------|----------|-----------------|
| Architecture | Unidirectional LSTM | BiLSTM + Attention |
| Parameters | Lower | ~40-50% more |
| Training Speed | Faster ⚡ | Slower (more compute) |
| Short Sequences | ✅ Good | ✅ Excellent |
| Long Sequences | ⚠️ Degrades | ✅ Maintains quality |
| Interpretability | ❌ Implicit | ✅ Attention weights visible |
| Translation Quality | Baseline | **Improved** 📈 |

### Generated Outputs

The training pipeline generates:
1. **Loss Curves** - Training/validation loss over epochs for both models
2. **Accuracy Metrics** - Token-level accuracy comparison
3. **Attention Heatmaps** - Visual interpretation of learned alignments
4. **Model Comparison Charts** - Side-by-side performance analysis

All results saved to `/results` directory for analysis and presentation.

## Key Insights

✅ **Why Attention Works:**
- Solves the **bottleneck problem** - doesn't compress all input into single vector
- Enables **dynamic focus** - decoder learns which encoder outputs matter
- Provides **interpretability** - attention weights show alignment
- Handles **long sequences** - maintains connection to earlier inputs

✅ **When to Use Each Model:**
- **Baseline**: Simple tasks, short sequences, educational purposes
- **Attention**: Production systems, longer sequences, interpretability needed

## Future Enhancements

- [ ] Implement multi-head attention (Transformer-style)
- [ ] Add self-attention mechanisms
- [ ] Implement beam search for better inference
- [ ] Calculate BLEU scores for automatic evaluation
- [ ] Support larger datasets (100k+ sentences)
- [ ] GPU optimization and distributed training
- [ ] Model checkpointing and resumable training
- [ ] Live translation inference mode
- [ ] Support for other language pairs

## References

1. **Bahdanau et al. (2014)** - Neural Machine Translation by Jointly Learning to Align and Translate
   - *Introduces the attention mechanism that powers the attention model*

2. **Sutskever et al. (2014)** - Sequence to Sequence Learning with Neural Networks
   - *Foundational work on encoder-decoder architectures*

3. **Vaswani et al. (2017)** - Attention Is All You Need
   - *Transformer architecture building on attention concepts*

## File Guide

| File | Purpose |
|------|---------|
| `dataset.py` | Data loading, vocabulary, preprocessing |
| `models.py` | Model architectures and components |
| `train_and_evaluate.py` | Training loop, evaluation, visualization |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Author & Acknowledgments

**Developer:** Amir Furquani (ID: 202301070165)

**Course:** Modelling and Deep Machine Learning (MDM)

**Project:** Lab Report on Encoder-Decoder Attention Mechanisms

---

## License

MIT License - Free to use for educational and research purposes.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{furquani2026encoderdecoder,
  author = {Furquani, Amir},
  title = {Encoder-Decoder Models with and without Attention},
  year = {2026},
  url = {https://github.com/Amirfurquani/202301070165_AMIR_Encoder-Decoder-Models-with-and-without-Attention}
}
```

---

**Note:** This project is designed for **educational purposes** to demonstrate the implementation and comparison of sequence-to-sequence models with and without attention mechanisms for machine translation tasks. For production use, consider pre-trained models like MarianMT or M2M models.
