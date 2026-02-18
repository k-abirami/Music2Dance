import librosa
import numpy as np


def extract_features(path: str, sr: int = 22050, n_mels: int = 128, max_duration: float = None):
    """
    Extract mel-spectrogram, beat frames, and onset envelope from audio.
    
    Args:
        path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel bands
        max_duration: Maximum duration in seconds to process (None = process all)
                     Useful for very long audio files

    Returns:
        mel (T, n_mels) float32
        beat_frames (F,) int frame indices aligned with mel frames
        onset_env (N,) float32 onset strength envelope
        y (samples,) float32 audio waveform
        sr (int) sample rate
    """
    print(f"Loading audio from {path}...")
    
    # Load audio, optionally limiting duration
    if max_duration is not None:
        y, sr = librosa.load(path, sr=sr, duration=max_duration)
        print(f"Loaded {max_duration}s of audio (or full file if shorter)")
    else:
        y, sr = librosa.load(path, sr=sr)
        duration = len(y) / sr
        print(f"Loaded {duration:.2f} seconds of audio")
        
        # Auto-limit very long files to prevent memory issues
        if duration > 60:  # If longer than 60 seconds
            print(f"File is very long ({duration:.2f}s). Limiting to first 60 seconds for performance.")
            max_samples = int(60 * sr)
            y = y[:max_samples]
            duration = 60

    print("Computing mel-spectrogram...")
    # mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    print("Detecting beats...")
    # onset strength + beat tracking in the same frame space
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # tempo can be a scalar or small ndarray depending on librosa version
    try:
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        tempo_str = f"{tempo_value:.1f}"
    except Exception:
        tempo_str = str(tempo)

    print(f"Detected {len(beat_frames)} beats at {tempo_str} BPM")
    print(f"Mel-spectrogram shape: {mel_db.T.shape}")

    return (
        mel_db.T.astype(np.float32),
        beat_frames.astype(np.int64),
        onset_env.astype(np.float32),
        y.astype(np.float32),
        sr,
    )

