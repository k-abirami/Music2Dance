# Music2Dance Code Explanation

## Overview

This project creates a 2D skeleton that dances to music using a combination of:
1. **Untrained ML models** (random initialization) for feature extraction
2. **Deterministic movement functions** that generate actual dance movements
3. **Audio feature extraction** using librosa

**Important Note**: The ML models are **NOT trained**. They use random weights initialized by PyTorch. The actual dancing comes from carefully designed deterministic functions that use audio features.

---

## Architecture Overview

```
Audio File (.wav)
    ↓
[Audio Feature Extraction] (librosa)
    ├─ Mel-spectrogram (T × 128)
    ├─ Beat frames
    └─ Onset envelope
    ↓
[MusicToDanceTransformer] (Untrained - Random Weights)
    └─ Audio embeddings (T × 256)
    ↓
[PrimitiveSelector] (Untrained - Random Weights)
    └─ Motion latents (T × 32)
    ↓
[MotionVAEStub] (Untrained - Random Weights)
    └─ Pose variations (T × 48 = 24 joints × 2 coords)
    ↓
[Base Skeleton Pose] (T-pose initialization)
    ↓
[generate_ml_dance_moves] (Deterministic Function)
    └─ Creative movement patterns based on audio features
    ↓
[Kinematic Constraints] (preserve_bone_lengths)
    ↓
Final Pose Sequence (T × 24 × 2)
```

---

## Component Breakdown

### 1. Audio Feature Extraction (`app/audio.py`)

**Purpose**: Extract musical features from audio files

**Process**:
```python
def extract_features(path, sr=22050, n_mels=128, max_duration=None):
    # 1. Load audio waveform
    y, sr = librosa.load(path, sr=sr, duration=max_duration)
    
    # 2. Compute mel-spectrogram (frequency representation)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to dB
    
    # 3. Detect beats and onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    return mel_db.T, beat_frames, onset_env, y, sr
```

**Outputs**:
- `mel_db.T`: (T, 128) - Mel-spectrogram frames over time
- `beat_frames`: Beat locations as frame indices
- `onset_env`: Onset strength envelope (rhythm intensity)
- `y`: Raw audio waveform
- `sr`: Sample rate

**Why mel-spectrogram?**
- Captures frequency content over time
- Similar to how humans perceive sound
- 128 mel bands = frequency resolution
- Each frame represents ~23ms of audio (at 22050 Hz)

---

### 2. MusicToDanceTransformer (`app/model.py`)

**Purpose**: Extract rich audio context embeddings (UNTRAINED)

**Architecture**:

```python
class MusicToDanceTransformer(nn.Module):
    def __init__(self, mel_dim=128, d_model=256):
        # 1. Audio projection: mel (128) → d_model (256)
        self.audio_proj = Linear(128 → 256) + LayerNorm + ReLU
        
        # 2. Positional encoding (sinusoidal)
        self.pos_encoding = PositionalEncoding(d_model=256)
        
        # 3. Transformer encoder (8 layers, 8 attention heads)
        self.transformer = TransformerEncoder(
            layers=8,
            heads=8,
            dim_feedforward=512
        )
        
        # 4. Multi-scale attention (2 heads)
        self.multi_scale_attn = [Attention1, Attention2]
        
        # 5. Beat-aware projection
        self.beat_proj = Linear(256 → 256) + LayerNorm + ReLU
        
        # 6. Temporal variation module
        self.temporal_variation = Linear(256 → 128 → 256)
```

**Forward Pass**:
```python
def forward(mel):
    # Input: (B, T, 128) mel-spectrogram
    
    # 1. Project to model dimension
    x = audio_proj(mel)  # (B, T, 256)
    
    # 2. Add positional encoding
    x = pos_encoding(x)  # Adds time position info
    
    # 3. Transformer encoding (8 layers)
    x = transformer(x)  # Self-attention across time
    
    # 4. Multi-scale attention
    x_short = multi_scale_attn[0](x, x, x)  # Short-term patterns
    x_long = multi_scale_attn[1](x, x, x)   # Long-term patterns
    x = x + 0.3*x_short + 0.2*x_long  # Combine
    
    # 5. Beat-aware transformation
    x = x + beat_proj(x)  # Residual connection
    
    # 6. Add temporal variation
    x = x + 0.1 * temporal_variation(x)  # Prevent repetition
    
    return x  # (B, T, 256)
```

**Key Design Choices**:
- **8 layers**: Deep enough to capture complex patterns
- **8 attention heads**: Multiple perspectives on audio
- **Multi-scale attention**: Captures both short and long-term patterns
- **Residual connections**: Helps with gradient flow (if trained)
- **Positional encoding**: Gives model sense of time

**Current State**: **Random weights** - outputs are essentially random but structured

---

### 3. PrimitiveSelector (`app/model.py`)

**Purpose**: Generate motion latents from audio context (UNTRAINED)

**Architecture**:

```python
class PrimitiveSelector(nn.Module):
    def __init__(self, d_model=256, latent_dim=32):
        # 1. Temporal self-attention
        self.temporal_attn = MultiheadAttention(256, heads=4)
        
        # 2. Policy network (deep)
        self.policy = Sequential(
            Linear(256 → 512) + LayerNorm + ReLU + Dropout(0.15),
            Linear(512 → 256) + LayerNorm + ReLU + Dropout(0.1),
            Linear(256 → 128) + ReLU,
            Linear(128 → 32)  # latent_dim
        )
        
        # 3. Variation head
        self.variation_head = Linear(256 → 128 → 32)
```

**Forward Pass**:
```python
def forward(context):
    # Input: (T, 256) audio context
    
    # 1. Temporal attention
    attn_out = temporal_attn(context, context, context)
    context_enhanced = context + 0.5 * attn_out
    
    # 2. Generate base latents
    base_latents = policy(context_enhanced)  # (T, 32)
    
    # 3. Add variation
    variation = variation_head(context_enhanced)  # (T, 32)
    latents = base_latents + 0.2 * variation
    
    return latents  # (T, 32)
```

**Purpose**: 
- Converts audio embeddings → motion latents
- Variation head adds randomness to prevent repetition
- **Currently**: Outputs random values (untrained)

---

### 4. MotionVAEStub (`app/model.py`)

**Purpose**: Decode latents to pose variations (UNTRAINED)

**Architecture**:

```python
class MotionVAEStub(nn.Module):
    def __init__(self, latent_dim=32, pose_dim=48):
        # 1. Transformer decoder (4 layers)
        self.temporal_decoder = TransformerDecoder(
            layers=4,
            heads=4,
            dim_feedforward=256
        )
        
        # 2. Pose decoder network
        self.decoder = Sequential(
            Linear(32 → 256) + LayerNorm + ReLU + Dropout,
            Linear(256 → 256) + LayerNorm + ReLU + Dropout,
            Linear(256 → 128) + ReLU,
            Linear(128 → 48)  # 24 joints × 2 coords
        )
        
        # 3. Learnable query embeddings
        self.query_embed = Parameter(1 × 32)
        
        # 4. Style embeddings (4 styles)
        self.style_embed = Parameter(4 × 32)
```

**Forward Pass**:
```python
def forward(z, style_idx=None):
    # Input: (T, 32) latents
    
    # 1. Create queries
    queries = query_embed.expand(T, -1)
    if style_idx:
        queries += 0.3 * style_embed[style_idx]
    
    # 2. Temporal decoding
    decoded = temporal_decoder(queries, z)  # (T, 32)
    
    # 3. Generate poses
    poses = decoder(decoded)  # (T, 48)
    
    return poses.reshape(T, 24, 2)  # (T, 24, 2)
```

**Purpose**:
- Converts motion latents → pose coordinates
- Transformer decoder ensures temporal smoothness
- **Currently**: Outputs small random variations

---

### 5. Movement Generation (`app/generate.py`)

**This is where the REAL dancing happens!**

The models above provide **subtle variations**, but the actual dance movements come from deterministic functions.

#### 5.1 Base Skeleton Initialization

```python
def initialize_skeleton_pose():
    # Creates T-pose with 24 joints
    # Joints: head, neck, spine, shoulders, elbows, wrists,
    #         pelvis, hips, knees, ankles, eyes, nose, etc.
    return pose  # (24, 2)
```

#### 5.2 Audio Feature Extraction for Motion

```python
def extract_audio_features_for_motion(audio_context, beat_frames, onset_env, T):
    # Extract multiple features from audio embeddings:
    features = {
        'intensity': mean(audio_context, axis=1),      # Overall energy
        'variation': std(audio_context, axis=1),       # Temporal variation
        'energy': mean(audio_context[:, :64]),         # High-frequency energy
        'rhythm': mean(audio_context[:, 64:128]),      # Rhythm features
        'beat_mask': create_beat_pulses(beat_frames),  # Beat locations
        'onset': normalize(onset_env)                  # Onset strength
    }
    return features
```

#### 5.3 Creative Dance Movement Generation

```python
def generate_ml_dance_moves(pose_seq, audio_context, beat_frames, onset_env, T):
    # Multi-phase system for complex motion
    time_phase = linspace(0, 12π, T)        # Normal phase
    time_phase_fast = linspace(0, 20π, T)   # Fast component
    time_phase_slow = linspace(0, 6π, T)    # Slow component
    
    # Dynamic dance style selection (0-3)
    dance_style = int((sin(style_phase) + 1) * 2) % 4
    
    for frame_idx in range(T):
        # Extract audio features
        intensity = features['intensity'][frame_idx]
        energy = features['energy'][frame_idx]
        rhythm = features['rhythm'][frame_idx]
        beat_strength = features['beat_mask'][frame_idx]
        
        # Calculate movement strength
        move_strength = (intensity*0.4 + energy*0.3 + rhythm*0.3) * (1 + beat_strength)
        
        # Multi-component phases
        current_phase = time_phase[frame_idx] + variation * 3π
        current_phase_fast = time_phase_fast[frame_idx] + variation * 5π
        
        # Apply movements based on dance style
        if dance_style == 0:  # Energetic
            # Wide swings, high kicks
        elif dance_style == 1:  # Smooth
            # Flowing, circular motions
        elif dance_style == 2:  # Quick
            # Sharp, fast movements
        else:  # Complex
            # Wave-like patterns
        
        # Apply to each body part:
        # - Head: bobbing patterns
        # - Arms: swinging with rotations
        # - Legs: stepping/kicking
        # - Torso: swaying
        # - Pelvis/Hips: rotation and tilt
```

**Key Features**:
- **4 dance styles** that switch dynamically
- **Multi-phase system** prevents repetition
- **Audio-driven** - movements respond to music
- **Kinematic constraints** preserve skeleton structure

#### 5.4 Kinematic Constraints

```python
def preserve_bone_lengths(pose, base_pose):
    # Ensures bones maintain correct lengths
    # Prevents skeleton from distorting
    
    for parent_idx, child_idx in bone_connections:
        base_length = ||base_pose[child] - base_pose[parent]||
        current_length = ||pose[child] - pose[parent]||
        
        # Scale to preserve length
        scale = base_length / current_length
        pose[child] = pose[parent] + (pose[child] - pose[parent]) * scale
    
    return pose
```

---

## Why This Works Without Training

The system works because:

1. **ML models provide variation**: Random weights create non-zero outputs that add subtle variation
2. **Deterministic functions do the work**: The actual dancing comes from `generate_ml_dance_moves()`
3. **Audio features guide movement**: Beat detection, onset strength, etc. drive the movements
4. **Multi-phase system**: Prevents repetition through complex phase combinations

---

## How to Actually Train This Model

To make this a **real ML system**, you would need:

### 1. Training Data

```python
# Need paired data:
# - Audio files
# - Corresponding motion capture data (skeleton poses)
# Format: (audio, pose_sequence) pairs
```

### 2. Loss Functions

```python
# Motion prediction loss
mse_loss = MSE(predicted_poses, ground_truth_poses)

# Temporal smoothness loss
smooth_loss = MSE(poses[t] - poses[t-1], target_velocity)

# Beat alignment loss
beat_loss = -log(probability of pose change at beat locations)

# Total loss
total_loss = mse_loss + 0.1*smooth_loss + 0.2*beat_loss
```

### 3. Training Loop

```python
def train(model, dataloader, optimizer, epochs=100):
    for epoch in range(epochs):
        for audio, poses_gt in dataloader:
            # Forward pass
            audio_features = extract_features(audio)
            context = audio_model(audio_features)
            latents = selector(context)
            poses_pred = motion_decoder(latents)
            
            # Calculate loss
            loss = mse_loss(poses_pred, poses_gt)
            loss += smoothness_loss(poses_pred)
            loss += beat_alignment_loss(poses_pred, beat_frames)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 4. Optimization

```python
# Optimizer
optimizer = torch.optim.AdamW(
    list(audio_model.parameters()) +
    list(selector.parameters()) +
    list(motion_decoder.parameters()),
    lr=1e-4,
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)

# Training tricks:
# - Gradient clipping
# - Mixed precision training
# - Data augmentation (time stretching, pitch shifting)
```

### 5. Evaluation Metrics

```python
# Motion quality metrics
fid_score = calculate_fid(predicted, ground_truth)
diversity_score = calculate_diversity(predicted_poses)
beat_alignment_score = calculate_beat_alignment(poses, beats)

# Perceptual metrics
human_evaluation = get_human_ratings(predicted_dances)
```

---

## Current System Strengths

1. **Works immediately** - No training data needed
2. **Creative movements** - 4 dynamic dance styles
3. **Beat-reactive** - Responds to music beats
4. **Non-repetitive** - Multi-phase system prevents loops
5. **Visually appealing** - Modern color scheme

## Current System Limitations

1. **Not learned** - Movements are hand-crafted, not learned from data
2. **Limited generalization** - Can't adapt to new dance styles without code changes
3. **No motion quality guarantee** - Random model outputs may create odd poses
4. **Deterministic** - Same audio → same dance (no stochasticity)

---

## File Structure

```
music2dance/
├── app/
│   ├── audio.py          # Audio feature extraction (librosa)
│   ├── model.py          # ML models (untrained)
│   ├── generate.py       # Movement generation (deterministic)
│   └── animate.py        # Visualization (matplotlib)
├── run_demo.py           # Main entry point
└── requirements.txt      # Dependencies
```

---

## Summary

**The ML models are sophisticated architectures but use random weights.**
- They provide structured variation
- The actual dancing comes from deterministic functions
- Audio features guide the movements
- The system works well despite no training!

**To make it truly ML-driven**, you'd need:
- Motion capture dataset
- Training loop with appropriate losses
- Optimization and hyperparameter tuning
- Evaluation on held-out data

The current system is a **hybrid approach** that combines:
- ML architecture (for structure)
- Deterministic functions (for quality)
- Audio features (for music synchronization)

This makes it work well without requiring expensive motion capture data!
