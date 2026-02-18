import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequences.
    Supports sequences longer than max_len by computing on-the-fly.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        length = x.size(1)
        if length <= self.max_len:
            return x + self.pe[:, :length]
        else:
            # For very long sequences, compute positional encoding on-the-fly
            position = torch.arange(0, length, dtype=torch.float32, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device)
                * (-math.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(length, self.d_model, dtype=torch.float32, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)


class MotionVAEStub(nn.Module):
    """
    Advanced motion decoder that generates structured skeleton poses.
    Uses transformer decoder architecture for temporal coherence and varied motion.
    Expanded for 24 joints (48 dimensions).
    """

    def __init__(self, latent_dim: int = 32, pose_dim: int = 48, hidden: int = 256):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        
        # Temporal transformer decoder for smooth motion with attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=hidden,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=4  # Deeper for more complex patterns
        )
        
        # Multi-scale pose decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, pose_dim),
        )
        
        # Learnable query embeddings for decoder (multiple for variety)
        self.query_embed = nn.Parameter(torch.randn(1, latent_dim))
        
        # Style embedding for different dance styles
        self.style_embed = nn.Parameter(torch.randn(4, latent_dim))  # 4 different styles

    def forward(self, z: torch.Tensor, style_idx: int = None) -> torch.Tensor:
        """
        z: (B, T, latent_dim) or (T, latent_dim)
        style_idx: Optional style index (0-3) for different dance styles
        returns: (B, T, pose_dim) or (T, pose_dim)
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)  # Add batch dimension
        
        B, T, _ = z.shape
        
        # Create query embeddings with optional style variation
        base_queries = self.query_embed.unsqueeze(0).expand(B, T, -1)
        
        # Add style variation if specified
        if style_idx is not None:
            style_vec = self.style_embed[style_idx % 4].unsqueeze(0).unsqueeze(0)
            queries = base_queries + style_vec * 0.3
        else:
            queries = base_queries
        
        # Temporal decoding for smooth transitions
        decoded = self.temporal_decoder(queries, z)
        
        # Generate poses
        poses = self.decoder(decoded)
        
        if z.dim() == 2:
            poses = poses.squeeze(0)
        
        return poses


class MusicToDanceTransformer(nn.Module):
    """
    Advanced transformer encoder that maps mel-spectrogram frames
    to rich audio context embeddings with beat-aware attention.
    """

    def __init__(self, mel_dim: int = 128, d_model: int = 256):
        super().__init__()

        # Multi-scale audio feature extraction
        self.audio_proj = nn.Sequential(
            nn.Linear(mel_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.pos_encoding = PositionalEncoding(d_model)

        # Deeper transformer with more heads for better audio understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)  # Deeper for more complexity
        
        # Multi-scale attention for capturing different temporal patterns
        self.multi_scale_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True),
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True),
        ])
        
        # Beat-aware projection with residual connection
        self.beat_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Temporal variation module to prevent repetition
        self.temporal_variation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, T, mel_dim)
        returns: (B, T, d_model)
        """
        x = self.audio_proj(mel)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        
        # Multi-scale attention for varied temporal patterns
        # Short-term patterns
        x_short, _ = self.multi_scale_attn[0](x, x, x)
        # Long-term patterns (with different key/query)
        x_long, _ = self.multi_scale_attn[1](x, x, x)
        
        # Combine multi-scale features
        x = x + 0.3 * x_short + 0.2 * x_long
        
        # Beat-aware transformation
        x = x + self.beat_proj(x)  # Residual connection
        
        # Add temporal variation to prevent repetition
        temporal_var = self.temporal_variation(x)
        x = x + 0.1 * temporal_var  # Small variation component
        
        return x


class PrimitiveSelector(nn.Module):
    """
    Chooses a latent motion code from the audio context at each frame.
    Uses attention mechanism and temporal context to focus on beat-relevant features
    and generate varied movements.
    """

    def __init__(self, d_model: int = 256, latent_dim: int = 32):
        super().__init__()

        # Self-attention for temporal context
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Motion policy network with more capacity
        self.policy = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
        # Variation head for non-repetitive motion
        self.variation_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, T, d_model) or (T, d_model)
        """
        if context.dim() == 2:
            context = context.unsqueeze(0)
        
        # Apply temporal attention for context-aware motion selection
        attn_out, _ = self.temporal_attn(context, context, context)
        context_enhanced = context + 0.5 * attn_out  # Residual connection
        
        # Generate base motion latents
        base_latents = self.policy(context_enhanced)
        
        # Add variation component for non-repetitive motion
        variation = self.variation_head(context_enhanced)
        latents = base_latents + 0.2 * variation  # Small variation component
        
        if context.dim() == 2:
            latents = latents.squeeze(0)
        
        return latents
