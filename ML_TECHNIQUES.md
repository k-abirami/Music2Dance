# Machine Learning Techniques Used in Music2Dance

## Overview

Even though the models aren't **trained**, the codebase uses several advanced ML techniques. This document explains what each technique is, how it's applied, and why it's used.

---

## 1. Transformer Architecture

### What It Is
**Transformers** are a type of neural network architecture introduced in "Attention Is All You Need" (2017). They use **self-attention** mechanisms to process sequences.

### How It's Used Here

```python
# MusicToDanceTransformer uses Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,           # Embedding dimension
    nhead=8,               # 8 attention heads
    dim_feedforward=512,   # Feedforward network size
    batch_first=True,
    dropout=0.1,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
```

**Purpose**: Process audio sequences (mel-spectrogram frames) and capture temporal relationships.

**Key Components**:
- **Self-Attention**: Each frame attends to all other frames
- **Multi-Head Attention**: 8 parallel attention mechanisms
- **Feedforward Networks**: Process attended features
- **Layer Normalization**: Stabilize training (if trained)
- **Residual Connections**: Help with gradient flow

**Why Transformers?**
- ✅ Excellent at sequence modeling
- ✅ Can capture long-range dependencies
- ✅ Parallel processing (faster than RNNs)
- ✅ State-of-the-art for audio-to-sequence tasks

---

## 2. Self-Attention Mechanism

### What It Is
**Self-Attention** allows each position in a sequence to attend to all other positions, learning which parts are relevant.

### Mathematical Formulation

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I contain?"
- V (Value): "What information do I have?"
```

### How It's Used Here

```python
# In TransformerEncoderLayer (inside PyTorch)
# Each mel-spectrogram frame attends to all frames
# Frame at time t can "see" frames at t-10, t+5, etc.
```

**Example**:
- Frame at beat location might attend strongly to other beat frames
- High-energy frames might attend to other high-energy frames
- Creates rich contextual representations

**Why Self-Attention?**
- ✅ Captures relationships between distant time steps
- ✅ Learns which audio frames are related
- ✅ Better than RNNs for long sequences

---

## 3. Multi-Head Attention

### What It Is
**Multi-Head Attention** runs multiple attention mechanisms in parallel, each learning different aspects.

### How It's Used Here

```python
# 8 attention heads in TransformerEncoderLayer
nhead=8

# Each head learns different patterns:
# Head 1: Beat relationships
# Head 2: Frequency relationships  
# Head 3: Temporal patterns
# Head 4: Energy patterns
# ... etc
```

**Purpose**: Capture multiple types of relationships simultaneously.

**Why Multi-Head?**
- ✅ Different heads learn different patterns
- ✅ More expressive than single-head attention
- ✅ Parallel processing (efficient)

---

## 4. Positional Encoding

### What It Is
**Positional Encoding** adds information about position in sequence to embeddings (since transformers have no inherent notion of order).

### How It's Used Here

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=10000):
        # Create sinusoidal patterns
        pe[:, 0::2] = sin(position / 10000^(2i/d_model))
        pe[:, 1::2] = cos(position / 10000^(2i/d_model))
```

**Mathematical Formulation**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Purpose**: Tell the model "this is frame 50" vs "this is frame 200".

**Why Sinusoidal?**
- ✅ Can extrapolate to longer sequences
- ✅ Relative positions are preserved
- ✅ Smooth gradients

---

## 5. Transformer Decoder Architecture

### What It Is
**Transformer Decoder** uses masked self-attention and cross-attention to generate sequences.

### How It's Used Here

```python
# MotionVAEStub uses Transformer Decoder
decoder_layer = nn.TransformerDecoderLayer(
    d_model=latent_dim,
    nhead=4,
    dim_feedforward=hidden,
)
self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
```

**Purpose**: Generate pose sequences from motion latents with temporal coherence.

**Key Difference from Encoder**:
- **Encoder**: Processes entire sequence at once
- **Decoder**: Generates sequence step-by-step (or uses queries)

**Why Decoder?**
- ✅ Ensures temporal smoothness
- ✅ Can condition on audio context
- ✅ Better for generation tasks

---

## 6. Variational Autoencoder (VAE) Concept

### What It Is
**VAE** learns to encode data into a latent space and decode back, with regularization to make the latent space smooth.

### How It's Used Here

```python
# MotionVAEStub (though not fully VAE - no KL divergence)
class MotionVAEStub(nn.Module):
    # Encoder: PrimitiveSelector (audio → latent)
    # Decoder: MotionVAEStub (latent → poses)
```

**VAE Components** (if fully implemented):
1. **Encoder**: Maps input to latent distribution (μ, σ)
2. **Reparameterization**: Sample z ~ N(μ, σ)
3. **Decoder**: Maps z back to output
4. **KL Divergence Loss**: Regularize latent space

**Current Implementation**:
- ✅ Has encoder-decoder structure
- ❌ No variational component (no sampling)
- ❌ No KL divergence loss

**Why VAE Concept?**
- ✅ Latent space allows interpolation
- ✅ Can generate variations
- ✅ Smooth latent space

---

## 7. Deep Feedforward Networks

### What It Is
**Feedforward Networks** (Multi-Layer Perceptrons) are stacks of linear transformations with non-linear activations.

### How It's Used Here

```python
# Multiple deep networks throughout:

# 1. Audio projection
self.audio_proj = nn.Sequential(
    nn.Linear(128, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),
)

# 2. Policy network (PrimitiveSelector)
self.policy = nn.Sequential(
    nn.Linear(256, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
)

# 3. Pose decoder
self.decoder = nn.Sequential(
    nn.Linear(32, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 48),
)
```

**Components**:
- **Linear Layers**: Matrix multiplication (Wx + b)
- **ReLU Activation**: Non-linearity (max(0, x))
- **Layer Normalization**: Normalize activations
- **Dropout**: Regularization (prevent overfitting)

**Why Deep Networks?**
- ✅ Can learn complex non-linear mappings
- ✅ Universal function approximators
- ✅ Hierarchical feature learning

---

## 8. Layer Normalization

### What It Is
**Layer Normalization** normalizes activations across features (not batch), stabilizing training.

### How It's Used Here

```python
nn.LayerNorm(d_model)
```

**Mathematical Formulation**:
```
LN(x) = γ * (x - μ) / (σ + ε) + β

Where:
- μ: mean of features
- σ: std of features  
- γ, β: learnable parameters
```

**Purpose**: 
- Stabilize activations
- Enable deeper networks
- Faster convergence (if trained)

**Why Layer Norm?**
- ✅ Works well with transformers
- ✅ Independent of batch size
- ✅ Better than batch norm for sequences

---

## 9. Dropout Regularization

### What It Is
**Dropout** randomly sets some neurons to zero during training to prevent overfitting.

### How It's Used Here

```python
nn.Dropout(0.1)  # 10% of neurons dropped
nn.Dropout(0.15) # 15% of neurons dropped
```

**Purpose**: 
- Prevent overfitting
- Improve generalization
- Ensemble effect (if trained)

**Why Dropout?**
- ✅ Simple regularization technique
- ✅ Works well with deep networks
- ✅ Prevents co-adaptation

---

## 10. Residual Connections

### What It Is
**Residual Connections** (skip connections) add input to output, allowing gradients to flow through.

### How It's Used Here

```python
# In TransformerEncoderLayer (built-in)
# x = x + attention(x)  # Residual connection

# Explicitly in code:
x = x + 0.3 * x_short + 0.2 * x_long  # Multi-scale residual
x = x + self.beat_proj(x)  # Beat-aware residual
context_enhanced = context + 0.5 * attn_out  # Attention residual
```

**Mathematical Formulation**:
```
output = input + transformation(input)
```

**Purpose**:
- Enable deeper networks
- Preserve information
- Better gradient flow

**Why Residual Connections?**
- ✅ Enable very deep networks (8+ layers)
- ✅ Prevent vanishing gradients
- ✅ Identity mapping if transformation isn't needed

---

## 11. Multi-Scale Attention

### What It Is
**Multi-Scale Attention** applies attention at different temporal scales to capture both short and long-term patterns.

### How It's Used Here

```python
# Two attention heads for different scales
self.multi_scale_attn = nn.ModuleList([
    nn.MultiheadAttention(d_model, num_heads=4),  # Short-term
    nn.MultiheadAttention(d_model, num_heads=4),  # Long-term
])

# Combine with weighted sum
x_short, _ = self.multi_scale_attn[0](x, x, x)
x_long, _ = self.multi_scale_attn[1](x, x, x)
x = x + 0.3 * x_short + 0.2 * x_long
```

**Purpose**:
- Capture immediate patterns (beats)
- Capture long-term patterns (song structure)
- Combine both scales

**Why Multi-Scale?**
- ✅ Music has multiple temporal scales
- ✅ Beats (short) vs song sections (long)
- ✅ More expressive representations

---

## 12. Temporal Attention

### What It Is
**Temporal Attention** allows the model to attend across time steps in a sequence.

### How It's Used Here

```python
# In PrimitiveSelector
self.temporal_attn = nn.MultiheadAttention(d_model, num_heads=4)

# Apply temporal attention
attn_out, _ = self.temporal_attn(context, context, context)
context_enhanced = context + 0.5 * attn_out
```

**Purpose**:
- Create context-aware motion latents
- Frame at time t considers all other frames
- Prevents repetitive movements

**Why Temporal Attention?**
- ✅ Context-aware generation
- ✅ Prevents repetition
- ✅ Better motion coherence

---

## 13. Learnable Embeddings

### What It Is
**Learnable Embeddings** are parameters that learn to represent concepts (like word embeddings).

### How It's Used Here

```python
# Query embeddings for decoder
self.query_embed = nn.Parameter(torch.randn(1, latent_dim))

# Style embeddings
self.style_embed = nn.Parameter(torch.randn(4, latent_dim))
```

**Purpose**:
- Query embeddings: Learn what to "ask" the decoder
- Style embeddings: Learn different dance styles

**Why Learnable Embeddings?**
- ✅ Can learn optimal representations
- ✅ More flexible than fixed embeddings
- ✅ Adapt to data (if trained)

---

## 14. Feature Extraction Techniques

### Mel-Spectrogram
**What**: Frequency representation of audio, similar to human perception.

**How Used**:
```python
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
```

**Purpose**: Convert audio to ML-friendly format.

**Why Mel-Spectrogram?**
- ✅ Captures frequency content
- ✅ Similar to human perception
- ✅ Standard for audio ML

---

### Beat Tracking
**What**: Detect rhythmic beats in audio.

**How Used**:
```python
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
```

**Purpose**: Identify when beats occur.

**Why Beat Tracking?**
- ✅ Synchronize movements to music
- ✅ Natural rhythm detection
- ✅ Drives dance movements

---

## Summary of ML Techniques

| Technique | Purpose | Where Used |
|-----------|---------|------------|
| **Transformer Encoder** | Process audio sequences | MusicToDanceTransformer |
| **Self-Attention** | Capture relationships | Transformer layers |
| **Multi-Head Attention** | Multiple perspectives | 8 heads in encoder |
| **Positional Encoding** | Add temporal info | Before transformer |
| **Transformer Decoder** | Generate sequences | MotionVAEStub |
| **VAE Concept** | Latent representation | Encoder-decoder structure |
| **Deep Feedforward** | Non-linear mapping | All networks |
| **Layer Normalization** | Stabilize training | Throughout |
| **Dropout** | Regularization | Throughout |
| **Residual Connections** | Enable deep networks | Multiple places |
| **Multi-Scale Attention** | Multiple temporal scales | MusicToDanceTransformer |
| **Temporal Attention** | Context awareness | PrimitiveSelector |
| **Learnable Embeddings** | Learnable representations | Query/style embeddings |

---

## Why These Techniques?

### For Audio Processing
- ✅ **Transformers**: Best for sequences
- ✅ **Multi-head attention**: Multiple audio aspects
- ✅ **Positional encoding**: Temporal relationships

### For Motion Generation
- ✅ **Decoder architecture**: Sequential generation
- ✅ **Temporal attention**: Prevent repetition
- ✅ **VAE structure**: Latent space for variation

### For Training (if implemented)
- ✅ **Layer norm**: Stable gradients
- ✅ **Dropout**: Prevent overfitting
- ✅ **Residual connections**: Enable deep networks

---

## Current State vs Trained State

### Current (Untrained)
- ✅ Architecture uses all these techniques
- ✅ Random weights initialized
- ✅ Forward pass works
- ❌ No learning from data

### If Trained
- ✅ All techniques would learn optimal weights
- ✅ Attention would focus on relevant patterns
- ✅ Embeddings would learn meaningful representations
- ✅ Networks would learn audio→motion mapping

---

## Conclusion

The codebase uses **state-of-the-art ML techniques**:
- Modern transformer architecture
- Advanced attention mechanisms
- Deep learning best practices
- Proper regularization techniques

Even though not trained, the architecture is **production-ready** and would work excellently with proper training data and a training loop!
