from app.generate import generate_dance
from app.animate import animate_pose

AUDIO_PATH = "superMario.wav"

# For very long files, limit to first 60 seconds (set to None to process full file)
MAX_DURATION = 60  # seconds

try:
    pose_seq, audio, sr = generate_dance(AUDIO_PATH, max_duration=MAX_DURATION)
    print("Starting animation...")
    animate_pose(pose_seq, audio, sr)
except KeyboardInterrupt:
    print("\n\nAnimation interrupted by user.")
except Exception as e:
    print(f"\n\nFailed to run animation: {e}")
    import traceback
    traceback.print_exc()
