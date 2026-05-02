# Encoder-Decoder Models with and without Attention

## Overview

This project implements Seq2Seq (Sequence-to-Sequence) encoder-decoder models for machine translation tasks, comparing two architectures:
- **Model 1**: Seq2Seq WITHOUT Attention (Baseline)
- **Model 2**: Seq2Seq WITH Bahdanau Attention

The project is part of the MDM Lab Report on "Encoder-Decoder Models and Attention Mechanisms" and demonstrates how attention mechanisms improve translation quality by allowing the decoder to focus on different parts of the input sequence.

## Project Structure

```
.
├── dataset.py                 # Data utilities and English-French parallel corpus
├── models.py                  # Encoder-Decoder model architectures
├── train_and_evaluate.py      # Training and evaluation script
└── README.md                  # This file
```

## Features

### 1. Dataset Module (`dataset.py`)
- **Curated English-French Parallel Corpus**: 100+ English-French sentence pairs
- **Vocabulary Building**: Automatic vocabulary construction from data
- **Tokenization**: Word-level tokenization with special tokens (PAD, SOS, EOS)
- **Data Loading**: PyTorch DataLoader compatibility for batch processing

### 2. Model Architectures (`models.py`)

#### Model 1: Seq2Seq WITHOUT Attention
```
Encoder (LSTM) → Fixed Context Vector → Decoder (LSTM) → Output
```
- **EncoderNoAttn**: Simple LSTM encoder
- **DecoderNoAttn**: LSTM decoder using only final encoder hidden state
- **Seq2SeqNoAttention**: Complete baseline model

#### Model 2: Seq2Seq WITH Bahdanau Attention
```
Encoder (BiLSTM) → Attention Mechanism → Decoder (LSTM) → Output
```
- **EncoderWithAttn**: Bidirectional LSTM encoder
- **Attention Layer**: Bahdanau additive attention mechanism
- **DecoderWithAttn**: LSTM decoder with attention context
- **Seq2SeqWithAttention**: Complete attention-based model

### 3. Training & Evaluation (`train_and_evaluate.py`)
- **Training Pipeline**: Complete training loop with teacher forcing
- **Evaluation Metrics**: 
  - Loss computation
  - Accuracy measurement
  - Per-token accuracy (excluding padding)
- **Visualization**: 
  - Training/validation loss curves
  - Model comparison plots
  - Attention weight heatmaps
- **Results Storage**: Metrics and visualizations saved to `/results` directory

## Hyperparameters

```python
EMB_DIM = 128          # Embedding dimension
HIDDEN_DIM = 256       # Hidden state dimension for baseline
ENC_HIDDEN = 128       # Encoder hidden dimension for attention model
N_LAYERS = 2           # Number of LSTM layers
DROPOUT = 0.3          # Dropout rate
BATCH_SIZE = 32        # Batch size
N_EPOCHS = 60          # Number of training epochs
LR = 0.001             # Learning rate
MAX_LEN = 20           # Maximum sequence length
```

## Installation

### Requirements
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
```

### Setup
```bash
# Clone the repository
git clone https://github.com/Amirfurquani/202301070165_AMIR_Encoder-Decoder-Models-with-and-without-Attention.git
cd encoder-decoder-models

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline
```bash
python train_and_evaluate.py
```

This will:
1. Load the English-French parallel corpus
2. Build vocabularies
3. Initialize both models
4. Train both models for specified epochs
5. Evaluate on test set
6. Generate comparison plots and metrics
7. Save results to `/results` directory

### Output Files
- `training_metrics.json` - Training/validation losses and accuracies
- `model_comparison.png` - Side-by-side model performance comparison
- `loss_curves.png` - Training and validation loss curves
- `attention_heatmaps.png` - Attention weight visualizations (attention model only)

## Model Comparison

### Key Differences

| Aspect | Without Attention | With Attention |
|--------|-------------------|-----------------|
| Encoder | Unidirectional LSTM | Bidirectional LSTM |
| Context | Fixed vector (last hidden state) | Dynamic context via attention |
| Parameters | Lower | Higher |
| Training Time | Faster | Slower |
| Translation Quality | Baseline | Improved |
| Long Sequences | Poor performance | Better handling |

### Expected Results
- **Baseline Model**: Better for short sequences, simpler
- **Attention Model**: Better BLEU scores, handles longer sequences, learns alignments

## Architecture Details

### Attention Mechanism (Bahdanau)
The attention mechanism computes a context vector for each decoder step:

$$\text{score}(h_t, \bar{s}_i) = v^T \tanh(W_1 h_t + W_2 \bar{s}_i)$$

$$\alpha_{t,i} = \frac{\exp(\text{score}(h_t, \bar{s}_i))}{\sum_j \exp(\text{score}(h_t, \bar{s}_j))}$$

$$c_t = \sum_i \alpha_{t,i} \bar{s}_i$$

Where:
- $h_t$ is decoder hidden state
- $\bar{s}_i$ are encoder outputs
- $c_t$ is the context vector
- $\alpha_{t,i}$ are attention weights

## Training Details

### Teacher Forcing
- **Ratio**: 0.5 during training, 0.0 during evaluation
- **Purpose**: Stabilizes training by feeding ground truth tokens at some steps

### Optimization
- **Optimizer**: Adam with learning rate 0.001
- **Criterion**: CrossEntropyLoss
- **Gradient Clipping**: Norm clipped at 1.0

## Results Analysis

The project generates:
1. **Loss Curves**: Visualization of training progress
2. **Accuracy Metrics**: Token-level accuracy comparison
3. **Attention Heatmaps**: Visual interpretation of attention weights
4. **Model Comparison**: Side-by-side performance evaluation

## Future Enhancements

- [ ] Implement multi-head attention
- [ ] Add self-attention (Transformer-based models)
- [ ] Implement beam search for inference
- [ ] Add BLEU score evaluation
- [ ] Support for larger datasets (real translation corpora)
- [ ] GPU optimization
- [ ] Model checkpointing and resumable training
- [ ] Inference mode for live translation

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate
- Sutskever, I., Vanhoucke, V., Le, Q., Kombrink, S., & Devin, M. (2014). Sequence to Sequence Learning with Neural Networks
- Vaswani, A., Shazeer, N., & Parmar, N. (2017). Attention Is All You Need

## Author

**Amir Furquani** (ID: 202301070165)

## License

MIT License - feel free to use this code for educational purposes.

## Acknowledgments

- Course: Modelling and Deep Machine Learning (MDM)
- Project: Lab Report on Encoder-Decoder Attention Mechanisms

---

**Note**: This project is designed for educational purposes to demonstrate the implementation and comparison of sequence-to-sequence models with and without attention mechanisms for machine translation tasks.
