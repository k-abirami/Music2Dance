import numpy as np
import torch

from .audio import extract_features
from .model import MusicToDanceTransformer, MotionVAEStub, PrimitiveSelector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Joint groups for different movement patterns (expanded skeleton)
JOINT_GROUPS = {
    'head': [0, 16, 17, 18],  # head, eyes, nose
    'spine': [1, 2, 22, 21],  # neck, upper_spine, mid_spine, pelvis_back
    'torso': [1, 2, 3, 6, 9, 10, 13],  # neck, spine, shoulders, pelvis, hips
    'left_arm': [3, 4, 5],  # left shoulder, elbow, wrist
    'right_arm': [6, 7, 8],  # right shoulder, elbow, wrist
    'pelvis': [9, 10, 13, 19, 20, 21],  # pelvis center, hips, hip fronts, pelvis back
    'left_leg': [10, 11, 12],  # left hip, knee, ankle
    'right_leg': [13, 14, 15],  # right hip, knee, ankle
}

# Bone structure: (parent_joint, child_joint, relative_length)
BONE_STRUCTURE = [
    (1, 0, 0.2),   # neck to head
    (0, 14, 0.05), (0, 15, 0.05), (0, 16, 0.03),  # head to eyes/nose
    (1, 2, 0.25),  # neck to left shoulder
    (1, 5, 0.25),  # neck to right shoulder
    (2, 3, 0.3),   # left shoulder to left elbow
    (3, 4, 0.3),   # left elbow to left wrist
    (5, 6, 0.3),   # right shoulder to right elbow
    (6, 7, 0.3),   # right elbow to right wrist
    (1, 8, 0.3),   # neck to left hip (via torso)
    (1, 11, 0.3),  # neck to right hip (via torso)
    (2, 8, 0.35),  # left shoulder to left hip
    (5, 11, 0.35), # right shoulder to right hip
    (8, 9, 0.4),   # left hip to left knee
    (9, 10, 0.4),  # left knee to left ankle
    (11, 12, 0.4), # right hip to right knee
    (12, 13, 0.4), # right knee to right ankle
]


def initialize_skeleton_pose() -> np.ndarray:
    """
    Initialize skeleton in a proper T-pose (standing pose) with enhanced hip articulation.
    Returns: (24, 2) array of joint positions in normalized coordinates
    """
    pose = np.zeros((24, 2), dtype=np.float32)
    
    # Scale factor for skeleton size
    scale = 0.8
    
    # Head (top center)
    pose[0] = [0.0, 0.6 * scale]  # head
    pose[16] = [-0.03 * scale, 0.63 * scale]  # left eye
    pose[17] = [0.03 * scale, 0.63 * scale]  # right eye
    pose[18] = [0.0, 0.64 * scale]  # nose
    
    # Neck and Spine
    pose[1] = [0.0, 0.4 * scale]  # neck
    pose[2] = [0.0, 0.25 * scale]  # upper_spine
    pose[22] = [0.0, 0.1 * scale]  # mid_spine
    pose[21] = [0.0, 0.0]  # pelvis_back
    
    # Pelvis center (between hips)
    pose[9] = [0.0, -0.05 * scale]  # pelvis_center
    
    # Shoulders (horizontal, wider)
    pose[3] = [-0.25 * scale, 0.3 * scale]  # left shoulder
    pose[6] = [0.25 * scale, 0.3 * scale]  # right shoulder
    
    # Arms (extended horizontally)
    pose[4] = [-0.45 * scale, 0.25 * scale]  # left elbow
    pose[5] = [-0.65 * scale, 0.2 * scale]  # left wrist
    pose[7] = [0.45 * scale, 0.25 * scale]  # right elbow
    pose[8] = [0.65 * scale, 0.2 * scale]  # right wrist
    
    # Hips with enhanced articulation
    pose[10] = [-0.12 * scale, -0.08 * scale]  # left_hip
    pose[13] = [0.12 * scale, -0.08 * scale]  # right_hip
    pose[19] = [-0.15 * scale, -0.1 * scale]  # left_hip_front (pelvis articulation)
    pose[20] = [0.15 * scale, -0.1 * scale]  # right_hip_front (pelvis articulation)
    
    # Legs (straight down)
    pose[11] = [-0.12 * scale, -0.5 * scale]  # left knee
    pose[12] = [-0.12 * scale, -0.9 * scale]  # left ankle
    pose[14] = [0.12 * scale, -0.5 * scale]  # right knee
    pose[15] = [0.12 * scale, -0.9 * scale]  # right ankle
    
    return pose


def smooth_motion(pose_seq: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Temporal smoothing with a moving average.
    pose_seq: (T, J, 2)
    """
    if window <= 1:
        return pose_seq

    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.copy(pose_seq)

    T, J, C = pose_seq.shape
    for j in range(J):
        for c in range(C):
            smoothed[:, j, c] = np.convolve(pose_seq[:, j, c], kernel, mode="same")

    return smoothed


def rotate_point_around_center(point: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a point around a center by given angle (radians)."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rel_point = point - center
    rotated = np.array([
        rel_point[0] * cos_a - rel_point[1] * sin_a,
        rel_point[0] * sin_a + rel_point[1] * cos_a
    ])
    return rotated + center


def extract_audio_features_for_motion(
    audio_context: np.ndarray,
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    T: int
) -> dict:
    """
    Extract rich audio features to drive varied dance movements.
    Returns features that vary over time to prevent repetition.
    """
    features = {}
    
    # Resample onset envelope to match sequence length
    if len(onset_env) != T:
        onset_indices = np.linspace(0, len(onset_env) - 1, T).astype(int)
        onset_norm = onset_env[onset_indices]
        onset_norm = (onset_norm - onset_norm.min()) / (onset_norm.max() - onset_norm.min() + 1e-6)
    else:
        onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-6)
    
    # Beat mask
    beat_mask = np.zeros(T, dtype=np.float32)
    if len(beat_frames) > 0:
        beat_indices = np.clip(beat_frames.astype(int), 0, T - 1)
        for idx in beat_indices:
            pulse_width = max(5, T // 40)
            start = max(0, idx - pulse_width)
            end = min(T, idx + pulse_width)
            pulse = np.exp(-np.abs(np.arange(start, end) - idx) / (pulse_width / 2))
            beat_mask[start:end] = np.maximum(beat_mask[start:end], pulse)
    
    # Audio context features (mean, std, spectral centroid-like)
    if audio_context.ndim == 2:
        features['intensity'] = np.mean(audio_context, axis=1)
        features['variation'] = np.std(audio_context, axis=1)
        # Use different dimensions for different movement aspects
        features['energy'] = np.abs(audio_context[:, :audio_context.shape[1]//4]).mean(axis=1)
        features['rhythm'] = np.abs(audio_context[:, audio_context.shape[1]//4:audio_context.shape[1]//2]).mean(axis=1)
    else:
        features['intensity'] = np.ones(T) * 0.5
        features['variation'] = np.ones(T) * 0.3
        features['energy'] = onset_norm
        features['rhythm'] = beat_mask
    
    # Normalize features
    for key in features:
        feat = features[key]
        if feat.max() > feat.min():
            features[key] = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
        else:
            features[key] = np.ones_like(feat) * 0.5
    
    features['beat_mask'] = beat_mask
    features['onset'] = onset_norm
    
    return features


def preserve_bone_lengths(pose: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """
    Ensure bone lengths are preserved by constraining joint positions.
    This maintains the skeleton structure while allowing movement.
    """
    corrected_pose = pose.copy()
    
    # Bone connections with their base lengths (expanded skeleton)
    bone_connections = [
        (1, 0, 'head'),      # neck to head
        (1, 2, 'upper_spine'),   # neck to upper spine
        (2, 22, 'mid_spine'),    # upper spine to mid spine
        (22, 21, 'pelvis_back'),  # mid spine to pelvis back
        (21, 9, 'pelvis_center'), # pelvis back to pelvis center
        (2, 3, 'left_shoulder'),   # upper spine to left shoulder
        (2, 6, 'right_shoulder'),  # upper spine to right shoulder
        (3, 4, 'left_upper_arm'),   # left shoulder to elbow
        (4, 5, 'left_forearm'),     # left elbow to wrist
        (6, 7, 'right_upper_arm'),  # right shoulder to elbow
        (7, 8, 'right_forearm'),    # right elbow to wrist
        (9, 10, 'left_hip'),       # pelvis center to left hip
        (9, 13, 'right_hip'),      # pelvis center to right hip
        (10, 19, 'left_hip_front'), # left hip to left hip front
        (13, 20, 'right_hip_front'), # right hip to right hip front
        (10, 11, 'left_thigh'),       # left hip to knee
        (11, 12, 'left_shin'),       # left knee to ankle
        (13, 14, 'right_thigh'),    # right hip to knee
        (14, 15, 'right_shin'),     # right knee to ankle
    ]
    
    # Calculate and preserve bone lengths
    for parent_idx, child_idx, bone_name in bone_connections:
        if parent_idx < len(base_pose) and child_idx < len(base_pose):
            # Get base bone vector and length
            base_bone_vec = base_pose[child_idx] - base_pose[parent_idx]
            base_length = np.linalg.norm(base_bone_vec)
            
            if base_length > 1e-6:  # Avoid division by zero
                # Get current bone vector
                current_bone_vec = corrected_pose[child_idx] - corrected_pose[parent_idx]
                current_length = np.linalg.norm(current_bone_vec)
                
                # Scale to preserve length
                if current_length > 1e-6:
                    scale = base_length / current_length
                    corrected_pose[child_idx] = corrected_pose[parent_idx] + current_bone_vec * scale
    
    return corrected_pose


def generate_ml_dance_moves(
    pose_seq: np.ndarray,
    audio_context: np.ndarray,
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate varied dance movements using ML-learned audio features.
    Creates non-repetitive, complex movements that adapt to music.
    Maintains skeleton structure through kinematic constraints.
    """
    enhanced_pose = pose_seq.copy()
    base_pose = initialize_skeleton_pose()  # Always use original base pose
    
    # Extract rich audio features
    audio_features = extract_audio_features_for_motion(audio_context, beat_frames, onset_env, T)
    
    # Time-varying parameters based on audio with multiple phase components
    time_phase = np.linspace(0, 12 * np.pi, T)  # Even longer period
    time_phase_fast = np.linspace(0, 20 * np.pi, T)  # Fast component
    time_phase_slow = np.linspace(0, 6 * np.pi, T)  # Slow component
    
    # Movement style selector that changes over time
    style_phase = np.linspace(0, 4 * np.pi, T)
    
    for frame_idx in range(T):
        t = frame_idx / max(T, 1)
        
        # Get audio-driven parameters
        intensity = audio_features['intensity'][frame_idx]
        variation = audio_features['variation'][frame_idx]
        energy = audio_features['energy'][frame_idx]
        rhythm = audio_features['rhythm'][frame_idx]
        beat_strength = audio_features['beat_mask'][frame_idx]
        
        # Combine features for movement strength
        move_strength = (intensity * 0.4 + energy * 0.3 + rhythm * 0.3) * (1 + beat_strength)
        
        # Multi-component phase system for complex, non-repetitive motion
        phase_mod = variation * 3 * np.pi
        phase_mod_fast = variation * 5 * np.pi
        current_phase = time_phase[frame_idx] + phase_mod
        current_phase_fast = time_phase_fast[frame_idx] + phase_mod_fast
        current_phase_slow = time_phase_slow[frame_idx] + variation * np.pi
        
        # Creative movement style selector - changes dynamically
        style_value = np.sin(style_phase[frame_idx] + intensity * np.pi)
        dance_style = int((style_value + 1) * 2) % 4  # 0-3 styles
        
        # Pattern selection with more variety
        pattern_selector = (energy * 0.4 + rhythm * 0.3 + intensity * 0.3 + 
                          np.sin(current_phase_slow) * 0.2) % 1.0
        
        # 1. HEAD: Creative, varied bobbing patterns based on dance style
        if dance_style == 0:  # Energetic head bobbing
            head_motion_y = (np.sin(current_phase * 2) + 
                           np.sin(current_phase_fast * 0.5) * 0.5) * move_strength * 0.12
            head_motion_x = np.cos(current_phase * 1.3) * move_strength * 0.08
        elif dance_style == 1:  # Smooth circular motion
            head_motion_y = np.sin(current_phase * 1.5) * np.cos(current_phase_slow * 0.7) * move_strength * 0.1
            head_motion_x = np.cos(current_phase * 1.5) * np.sin(current_phase_slow * 0.7) * move_strength * 0.08
        elif dance_style == 2:  # Quick jerky movements
            head_motion_y = np.sin(current_phase_fast * 1.2) * move_strength * 0.15
            head_motion_x = np.cos(current_phase_fast * 1.1) * move_strength * 0.1
        else:  # Style 3: Wave-like motion
            head_motion_y = (np.sin(current_phase * 2.5) * np.cos(current_phase_slow) + 
                           np.sin(current_phase_fast * 0.8)) * move_strength * 0.1
            head_motion_x = np.cos(current_phase * 2.2) * move_strength * 0.06
        
        for j in JOINT_GROUPS['head']:
            if j < len(base_pose):
                enhanced_pose[frame_idx, j, 1] += head_motion_y
                enhanced_pose[frame_idx, j, 0] += head_motion_x
        
        # 2. ARMS: Creative swinging patterns that vary by dance style
        arm_phase_left = current_phase + np.pi * (0.3 + variation * 0.6)
        arm_phase_right = current_phase + np.pi * (0.7 + variation * 0.6)
        arm_phase_fast_left = current_phase_fast + np.pi * (0.2 + variation * 0.5)
        arm_phase_fast_right = current_phase_fast + np.pi * (0.8 + variation * 0.5)
        
        # Different arm patterns based on dance style
        if dance_style == 0:  # Wide, energetic swings
            left_swing = (np.sin(arm_phase_left) + np.sin(arm_phase_fast_left * 0.3) * 0.4) * move_strength * 0.6
            right_swing = -(np.sin(arm_phase_right) + np.sin(arm_phase_fast_right * 0.3) * 0.4) * move_strength * 0.6
            left_lift = np.cos(arm_phase_left * 0.8) * move_strength * 0.35
            right_lift = np.cos(arm_phase_right * 0.8) * move_strength * 0.35
        elif dance_style == 1:  # Smooth, flowing movements
            left_swing = np.sin(arm_phase_left * 0.7) * np.cos(current_phase_slow * 0.5) * move_strength * 0.45
            right_swing = -np.sin(arm_phase_right * 0.7) * np.cos(current_phase_slow * 0.5) * move_strength * 0.45
            left_lift = np.cos(arm_phase_left * 0.6) * move_strength * 0.25
            right_lift = np.cos(arm_phase_right * 0.6) * move_strength * 0.25
        elif dance_style == 2:  # Quick, sharp movements
            left_swing = np.sin(arm_phase_fast_left * 1.5) * move_strength * 0.5
            right_swing = -np.sin(arm_phase_fast_right * 1.5) * move_strength * 0.5
            left_lift = np.cos(arm_phase_fast_left * 1.2) * move_strength * 0.3
            right_lift = np.cos(arm_phase_fast_right * 1.2) * move_strength * 0.3
        else:  # Style 3: Complex wave patterns
            left_swing = (np.sin(arm_phase_left * 1.2) * np.cos(arm_phase_fast_left * 0.4) + 
                         np.sin(current_phase_slow * 0.8)) * move_strength * 0.5
            right_swing = -(np.sin(arm_phase_right * 1.2) * np.cos(arm_phase_fast_right * 0.4) + 
                           np.sin(current_phase_slow * 0.8)) * move_strength * 0.5
            left_lift = np.cos(arm_phase_left * 0.9) * np.sin(current_phase_slow * 0.6) * move_strength * 0.3
            right_lift = np.cos(arm_phase_right * 0.9) * np.sin(current_phase_slow * 0.6) * move_strength * 0.3
        
        # Apply arm rotations with varying angles
        # Use current frame's shoulder position (which may have moved with torso)
        left_shoulder = enhanced_pose[frame_idx, 3]  # Updated index
        right_shoulder = enhanced_pose[frame_idx, 6]  # Updated index
        
        # Get base arm positions relative to base shoulder
        base_left_elbow_rel = base_pose[4] - base_pose[3]  # Updated indices
        base_left_wrist_rel = base_pose[5] - base_pose[3]
        base_right_elbow_rel = base_pose[7] - base_pose[6]
        base_right_wrist_rel = base_pose[8] - base_pose[6]
        
        # Left arm - rotate around current shoulder position
        arm_angle_left = left_swing * (0.4 + variation * 0.2)  # Reduced max angle
        cos_a, sin_a = np.cos(arm_angle_left), np.sin(arm_angle_left)
        rotated_left_elbow_rel = np.array([
            base_left_elbow_rel[0] * cos_a - base_left_elbow_rel[1] * sin_a,
            base_left_elbow_rel[0] * sin_a + base_left_elbow_rel[1] * cos_a
        ])
        enhanced_pose[frame_idx, 3] = left_shoulder + rotated_left_elbow_rel
        enhanced_pose[frame_idx, 3, 1] += left_lift * 0.5  # Reduced lift
        
        cos_a2, sin_a2 = np.cos(arm_angle_left * 1.2), np.sin(arm_angle_left * 1.2)
        rotated_left_wrist_rel = np.array([
            base_left_wrist_rel[0] * cos_a2 - base_left_wrist_rel[1] * sin_a2,
            base_left_wrist_rel[0] * sin_a2 + base_left_wrist_rel[1] * cos_a2
        ])
        enhanced_pose[frame_idx, 4] = left_shoulder + rotated_left_wrist_rel
        enhanced_pose[frame_idx, 4, 1] += left_lift * 0.6
        
        # Right arm - rotate around current shoulder position
        arm_angle_right = right_swing * (0.4 + variation * 0.2)
        cos_a, sin_a = np.cos(arm_angle_right), np.sin(arm_angle_right)
        rotated_right_elbow_rel = np.array([
            base_right_elbow_rel[0] * cos_a - base_right_elbow_rel[1] * sin_a,
            base_right_elbow_rel[0] * sin_a + base_right_elbow_rel[1] * cos_a
        ])
        enhanced_pose[frame_idx, 6] = right_shoulder + rotated_right_elbow_rel
        enhanced_pose[frame_idx, 6, 1] += right_lift * 0.5
        
        cos_a2, sin_a2 = np.cos(arm_angle_right * 1.2), np.sin(arm_angle_right * 1.2)
        rotated_right_wrist_rel = np.array([
            base_right_wrist_rel[0] * cos_a2 - base_right_wrist_rel[1] * sin_a2,
            base_right_wrist_rel[0] * sin_a2 + base_right_wrist_rel[1] * cos_a2
        ])
        enhanced_pose[frame_idx, 7] = right_shoulder + rotated_right_wrist_rel
        enhanced_pose[frame_idx, 7, 1] += right_lift * 0.6
        
        # 3. LEGS: Creative stepping/kicking patterns based on dance style
        leg_phase = current_phase * (0.8 + energy * 0.5)
        leg_phase_fast = current_phase_fast * (1.2 + rhythm * 0.3)
        
        if dance_style == 0:  # Energetic kicks and jumps
            if beat_strength > 0.4:
                kick_strength = beat_strength * move_strength * 1.2
                leg_alternate = int(np.sin(leg_phase) > 0)
                if leg_alternate:
                    enhanced_pose[frame_idx, 11, 1] += kick_strength * 0.5  # left knee
                    enhanced_pose[frame_idx, 12, 1] += kick_strength * 0.4  # left ankle
                    enhanced_pose[frame_idx, 11, 0] += np.sin(leg_phase_fast) * kick_strength * 0.25
                else:
                    enhanced_pose[frame_idx, 14, 1] += kick_strength * 0.5  # right knee
                    enhanced_pose[frame_idx, 15, 1] += kick_strength * 0.4  # right ankle
                    enhanced_pose[frame_idx, 14, 0] += np.sin(leg_phase_fast + np.pi) * kick_strength * 0.25
            else:
                step_pattern = np.sin(leg_phase) * move_strength * 0.3
                enhanced_pose[frame_idx, 11, 1] += max(0, step_pattern) * 0.35
                enhanced_pose[frame_idx, 14, 1] += max(0, -step_pattern) * 0.35
        elif dance_style == 1:  # Smooth stepping
            step_pattern = np.sin(leg_phase * 0.7) * np.cos(current_phase_slow * 0.5) * move_strength * 0.25
            enhanced_pose[frame_idx, 11, 1] += max(0, step_pattern) * 0.3
            enhanced_pose[frame_idx, 14, 1] += max(0, -step_pattern) * 0.3
        elif dance_style == 2:  # Quick shuffling
            shuffle_pattern = np.sin(leg_phase_fast * 1.5) * move_strength * 0.2
            enhanced_pose[frame_idx, 11, 1] += shuffle_pattern * 0.25
            enhanced_pose[frame_idx, 14, 1] += -shuffle_pattern * 0.25
            enhanced_pose[frame_idx, 12, 0] += np.cos(leg_phase_fast) * move_strength * 0.15
            enhanced_pose[frame_idx, 15, 0] += -np.cos(leg_phase_fast) * move_strength * 0.15
        else:  # Style 3: Complex leg movements
            leg_motion = (np.sin(leg_phase * 1.1) * np.cos(leg_phase_fast * 0.3) + 
                        np.sin(current_phase_slow * 0.7)) * move_strength * 0.3
            enhanced_pose[frame_idx, 11, 1] += max(0, leg_motion) * 0.3
            enhanced_pose[frame_idx, 14, 1] += max(0, -leg_motion) * 0.3
            enhanced_pose[frame_idx, 12, 0] += np.sin(leg_phase * 0.9) * move_strength * 0.2
            enhanced_pose[frame_idx, 15, 0] += -np.sin(leg_phase * 0.9) * move_strength * 0.2
        
        # 4. TORSO/SPINE: Creative swaying patterns based on dance style
        if dance_style == 0:  # Energetic torso movement
            torso_sway_x = (np.sin(current_phase * 0.8) + np.sin(current_phase_fast * 0.4) * 0.5) * move_strength * 0.15
            torso_sway_y = np.cos(current_phase * 1.3) * move_strength * 0.1
        elif dance_style == 1:  # Smooth, flowing torso
            torso_sway_x = np.sin(current_phase * 0.5) * np.cos(current_phase_slow * 0.6) * move_strength * 0.12
            torso_sway_y = np.cos(current_phase * 1.0) * np.sin(current_phase_slow * 0.4) * move_strength * 0.08
        elif dance_style == 2:  # Quick, sharp torso movements
            torso_sway_x = np.sin(current_phase_fast * 1.2) * move_strength * 0.12
            torso_sway_y = np.cos(current_phase_fast * 1.0) * move_strength * 0.08
        else:  # Style 3: Complex wave-like torso
            torso_sway_x = (np.sin(current_phase * 0.7) * np.cos(current_phase_fast * 0.3) + 
                          np.sin(current_phase_slow * 0.5)) * move_strength * 0.13
            torso_sway_y = (np.cos(current_phase * 1.1) * np.sin(current_phase_slow * 0.6)) * move_strength * 0.09
        
        for j in JOINT_GROUPS['spine']:
            if j < len(base_pose):
                # Spine joints follow with decreasing influence
                influence = 1.0 - (JOINT_GROUPS['spine'].index(j) * 0.2)
                enhanced_pose[frame_idx, j, 0] += torso_sway_x * influence
                enhanced_pose[frame_idx, j, 1] += torso_sway_y * influence
        
        # 5. PELVIS/HIPS: Creative hip movement patterns based on dance style
        pelvis_center = enhanced_pose[frame_idx, 9]
        
        # Pelvis rotation/tilt varies by dance style
        if dance_style == 0:  # Energetic hip movements
            pelvis_rotation = (np.sin(current_phase * 0.9 + rhythm * np.pi) + 
                             np.sin(current_phase_fast * 0.3) * 0.4) * move_strength * 0.18
            pelvis_tilt_x = np.cos(current_phase * 1.1 + energy * np.pi) * move_strength * 0.12
            pelvis_tilt_y = np.sin(current_phase * 1.3) * move_strength * 0.1
        elif dance_style == 1:  # Smooth, flowing hips
            pelvis_rotation = np.sin(current_phase * 0.6) * np.cos(current_phase_slow * 0.7) * move_strength * 0.15
            pelvis_tilt_x = np.cos(current_phase * 0.8) * np.sin(current_phase_slow * 0.5) * move_strength * 0.1
            pelvis_tilt_y = np.sin(current_phase * 1.0) * move_strength * 0.08
        elif dance_style == 2:  # Quick, sharp hip movements
            pelvis_rotation = np.sin(current_phase_fast * 1.4) * move_strength * 0.16
            pelvis_tilt_x = np.cos(current_phase_fast * 1.2) * move_strength * 0.11
            pelvis_tilt_y = np.sin(current_phase_fast * 1.0) * move_strength * 0.09
        else:  # Style 3: Complex hip patterns
            pelvis_rotation = (np.sin(current_phase * 0.8) * np.cos(current_phase_fast * 0.4) + 
                             np.sin(current_phase_slow * 0.6)) * move_strength * 0.17
            pelvis_tilt_x = (np.cos(current_phase * 1.0) * np.sin(current_phase_slow * 0.5) + 
                           np.cos(current_phase_fast * 0.3)) * move_strength * 0.11
            pelvis_tilt_y = np.sin(current_phase * 1.2) * np.cos(current_phase_slow * 0.4) * move_strength * 0.09
        
        # Move pelvis center
        enhanced_pose[frame_idx, 9, 0] += pelvis_tilt_x
        enhanced_pose[frame_idx, 9, 1] += pelvis_tilt_y
        
        # Pelvis back (spine connection) follows pelvis center with slight delay
        enhanced_pose[frame_idx, 21, 0] += pelvis_tilt_x * 0.8
        enhanced_pose[frame_idx, 21, 1] += pelvis_tilt_y * 0.8
        
        # Left hip - complex movement
        left_hip_base = base_pose[10] - base_pose[9]
        left_hip_angle = pelvis_rotation + np.sin(current_phase * 0.6) * move_strength * 0.2
        cos_lh, sin_lh = np.cos(left_hip_angle), np.sin(left_hip_angle)
        rotated_left_hip = np.array([
            left_hip_base[0] * cos_lh - left_hip_base[1] * sin_lh,
            left_hip_base[0] * sin_lh + left_hip_base[1] * cos_lh
        ])
        enhanced_pose[frame_idx, 10] = pelvis_center + rotated_left_hip
        
        # Left hip front (pelvis articulation point)
        left_hip_front_base = base_pose[19] - base_pose[10]
        hip_front_angle = left_hip_angle * 0.5 + np.cos(current_phase * 0.8) * move_strength * 0.15
        cos_lhf, sin_lhf = np.cos(hip_front_angle), np.sin(hip_front_angle)
        rotated_left_hip_front = np.array([
            left_hip_front_base[0] * cos_lhf - left_hip_front_base[1] * sin_lhf,
            left_hip_front_base[0] * sin_lhf + left_hip_front_base[1] * cos_lhf
        ])
        enhanced_pose[frame_idx, 19] = enhanced_pose[frame_idx, 10] + rotated_left_hip_front
        
        # Right hip - complex movement (opposite phase)
        right_hip_base = base_pose[13] - base_pose[9]
        right_hip_angle = -pelvis_rotation + np.sin(current_phase * 0.6 + np.pi) * move_strength * 0.2
        cos_rh, sin_rh = np.cos(right_hip_angle), np.sin(right_hip_angle)
        rotated_right_hip = np.array([
            right_hip_base[0] * cos_rh - right_hip_base[1] * sin_rh,
            right_hip_base[0] * sin_rh + right_hip_base[1] * cos_rh
        ])
        enhanced_pose[frame_idx, 13] = pelvis_center + rotated_right_hip
        
        # Right hip front (pelvis articulation point)
        right_hip_front_base = base_pose[20] - base_pose[13]
        hip_front_angle_r = right_hip_angle * 0.5 + np.cos(current_phase * 0.8 + np.pi) * move_strength * 0.15
        cos_rhf, sin_rhf = np.cos(hip_front_angle_r), np.sin(hip_front_angle_r)
        rotated_right_hip_front = np.array([
            right_hip_front_base[0] * cos_rhf - right_hip_front_base[1] * sin_rhf,
            right_hip_front_base[0] * sin_rhf + right_hip_front_base[1] * cos_rhf
        ])
        enhanced_pose[frame_idx, 20] = enhanced_pose[frame_idx, 13] + rotated_right_hip_front
        
        # Spine joints follow pelvis movement with decreasing influence
        enhanced_pose[frame_idx, 22, 0] += pelvis_tilt_x * 0.6  # mid_spine
        enhanced_pose[frame_idx, 22, 1] += pelvis_tilt_y * 0.6
        enhanced_pose[frame_idx, 2, 0] += pelvis_tilt_x * 0.3  # upper_spine
        enhanced_pose[frame_idx, 2, 1] += pelvis_tilt_y * 0.3
        
        # Preserve bone lengths after each frame's modifications
        enhanced_pose[frame_idx] = preserve_bone_lengths(
            enhanced_pose[frame_idx], 
            base_pose
        )
    
    return enhanced_pose


def apply_beat_reactive_motion(
    pose_seq: np.ndarray,
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    mel_frames: int
) -> np.ndarray:
    """
    Apply natural, human-like dance movements that react to beats.
    Preserves skeleton structure while creating realistic motion.
    """
    T, J, C = pose_seq.shape
    enhanced_pose = pose_seq.copy()
    
    if len(beat_frames) == 0:
        return enhanced_pose
    
    # Normalize onset envelope
    if len(onset_env) > 0:
        onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-6)
        if len(onset_norm) != T:
            onset_indices = np.linspace(0, len(onset_norm) - 1, T).astype(int)
            onset_norm = onset_norm[onset_indices]
    else:
        onset_norm = np.ones(T) * 0.5
    
    # Create beat mask with smooth pulses
    beat_mask = np.zeros(T, dtype=np.float32)
    beat_indices = np.clip(beat_frames.astype(int), 0, T - 1)
    
    for idx in beat_indices:
        pulse_width = max(5, T // 40)
        start = max(0, idx - pulse_width)
        end = min(T, idx + pulse_width)
        pulse = np.exp(-np.abs(np.arange(start, end) - idx) / (pulse_width / 2))
        beat_mask[start:end] = np.maximum(beat_mask[start:end], pulse)
    
    # Time-based phase for continuous motion
    time_phase = np.linspace(0, 4 * np.pi, T)
    
    for frame_idx in range(T):
        t = frame_idx / max(T, 1)  # Normalized time [0, 1]
        beat_strength = beat_mask[frame_idx]
        music_intensity = onset_norm[frame_idx]
        combined_strength = beat_strength * music_intensity
        
        # Base pose reference
        base = pose_seq[frame_idx].copy()
        
        # 1. HEAD: Bobbing motion synchronized to beats
        head_bob = np.sin(time_phase[frame_idx] * 2) * beat_strength * 0.08
        for j in JOINT_GROUPS['head']:
            if j < J:
                enhanced_pose[frame_idx, j, 1] += head_bob
        
        # 2. ARMS: Natural swinging motion (like walking/dancing)
        # Left arm swings forward/backward
        arm_phase = time_phase[frame_idx] + np.pi * 0.3  # Slight phase offset
        left_arm_swing = np.sin(arm_phase) * combined_strength * 0.4
        
        # Rotate left arm around shoulder
        left_shoulder = base[2]  # left shoulder index
        left_elbow_base = base[3]
        left_wrist_base = base[4]
        
        # Calculate rotation angle for arm swing
        arm_angle = left_arm_swing * 0.6  # Max 0.6 radians (~34 degrees)
        enhanced_pose[frame_idx, 3] = rotate_point_around_center(
            left_elbow_base, left_shoulder, arm_angle
        )
        enhanced_pose[frame_idx, 4] = rotate_point_around_center(
            left_wrist_base, left_shoulder, arm_angle * 1.2
        )
        
        # Right arm swings opposite phase
        right_arm_swing = -np.sin(arm_phase) * combined_strength * 0.4
        right_shoulder = base[5]
        right_elbow_base = base[6]
        right_wrist_base = base[7]
        
        arm_angle_right = right_arm_swing * 0.6
        enhanced_pose[frame_idx, 6] = rotate_point_around_center(
            right_elbow_base, right_shoulder, arm_angle_right
        )
        enhanced_pose[frame_idx, 7] = rotate_point_around_center(
            right_wrist_base, right_shoulder, arm_angle_right * 1.2
        )
        
        # On strong beats, lift arms up
        if beat_strength > 0.5:
            lift_amount = (beat_strength - 0.5) * 0.3
            enhanced_pose[frame_idx, 3, 1] += lift_amount
            enhanced_pose[frame_idx, 4, 1] += lift_amount * 1.2
            enhanced_pose[frame_idx, 6, 1] += lift_amount
            enhanced_pose[frame_idx, 7, 1] += lift_amount * 1.2
        
        # 3. LEGS: Alternating step/kick motion
        leg_phase = time_phase[frame_idx]
        left_leg_lift = np.sin(leg_phase) * combined_strength * 0.25
        right_leg_lift = -np.sin(leg_phase) * combined_strength * 0.25
        
        # Left leg: lift knee and ankle
        if left_leg_lift > 0:
            enhanced_pose[frame_idx, 9, 1] += left_leg_lift * 0.3  # knee up
            enhanced_pose[frame_idx, 10, 1] += left_leg_lift * 0.2  # ankle follows
            enhanced_pose[frame_idx, 9, 0] += left_leg_lift * 0.15  # slight forward
        
        # Right leg: lift knee and ankle
        if right_leg_lift > 0:
            enhanced_pose[frame_idx, 12, 1] += right_leg_lift * 0.3
            enhanced_pose[frame_idx, 13, 1] += right_leg_lift * 0.2
            enhanced_pose[frame_idx, 12, 0] += right_leg_lift * 0.15
        
        # On strong beats, bigger leg movements
        if beat_strength > 0.6:
            kick_strength = (beat_strength - 0.6) * 0.4
            # Alternate kicks
            if frame_idx % 2 == 0:
                enhanced_pose[frame_idx, 9, 1] += kick_strength * 0.2
                enhanced_pose[frame_idx, 10, 1] += kick_strength * 0.15
            else:
                enhanced_pose[frame_idx, 12, 1] += kick_strength * 0.2
                enhanced_pose[frame_idx, 13, 1] += kick_strength * 0.15
        
        # 4. TORSO: Subtle swaying and vertical bounce
        torso_sway = np.sin(time_phase[frame_idx] * 0.5) * beat_strength * 0.08
        torso_bounce = np.sin(time_phase[frame_idx] * 3) * beat_strength * 0.05
        
        for j in JOINT_GROUPS['torso']:
            if j < J:
                enhanced_pose[frame_idx, j, 0] += torso_sway
                enhanced_pose[frame_idx, j, 1] += torso_bounce
        
        # 5. HIPS: Slight rotation/rocking
        hip_rock = np.sin(time_phase[frame_idx] * 0.7) * combined_strength * 0.06
        enhanced_pose[frame_idx, 8, 0] += hip_rock
        enhanced_pose[frame_idx, 11, 0] -= hip_rock
    
    return enhanced_pose


audio_model = MusicToDanceTransformer().to(DEVICE)
selector = PrimitiveSelector().to(DEVICE)
motion_decoder = MotionVAEStub().to(DEVICE)

audio_model.eval()
selector.eval()
motion_decoder.eval()


def generate_dance(audio_path: str, max_duration: float = None):
    """
    Full audio → pose sequence pipeline with natural human-like dance motion.
    Maintains proper skeleton structure throughout.
    
    Args:
        audio_path: Path to audio file
        max_duration: Maximum duration in seconds to process (None = process all)
    """
    try:
        print(f"\n{'='*50}")
        print(f"Generating dance for: {audio_path}")
        print(f"{'='*50}\n")
        
        mel, beat_frames, onset_env, y, sr = extract_features(audio_path, max_duration=max_duration)
        
        # Initialize base skeleton pose (proper T-pose)
        base_pose = initialize_skeleton_pose()
        T = mel.shape[0]
        
        print(f"Generating {T} pose frames...")
        
        # Create pose sequence starting from base pose (24 joints now)
        pose_seq = np.tile(base_pose[np.newaxis, :, :], (T, 1, 1)).astype(np.float32)
        
        # Use audio features to modulate motion intensity
        print("Processing with transformer model...")
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Get rich audio context embeddings
            context = audio_model(mel_tensor)[0]  # (T, d_model)
            
            # Generate motion latents from audio context
            latent_seq = selector(context)  # (T, latent_dim)
            
            # Use audio features to select dance style (varies over time)
            audio_intensity = context.mean(dim=1).cpu().numpy()  # (T,)
            audio_intensity_norm = (audio_intensity - audio_intensity.min()) / (audio_intensity.max() - audio_intensity.min() + 1e-6)
            
            # Generate pose variations from latents (this creates subtle motion structure)
            pose_variations = motion_decoder(latent_seq).cpu().numpy()  # (T, 48)
            pose_variations = pose_variations.reshape(-1, 24, 2)
            
            # Normalize variations to very small scale to preserve structure
            pose_std = np.std(pose_variations, axis=(0, 1), keepdims=True)
            pose_mean = np.mean(pose_variations, axis=(0, 1), keepdims=True)
            pose_variations = (pose_variations - pose_mean) / (pose_std + 1e-6) * 0.05  # Much smaller scale
            
            # Add subtle variations to base pose (only for subtle motion hints)
            pose_seq = pose_seq + pose_variations
            
            # Immediately preserve structure after adding variations
            for t in range(T):
                pose_seq[t] = preserve_bone_lengths(pose_seq[t], base_pose)
        
        print("Generating ML-driven dance movements...")
        # Apply sophisticated ML-based dance movements using audio context
        pose_seq = generate_ml_dance_moves(
            pose_seq, 
            context.cpu().numpy(),  # Full audio context for feature extraction
            beat_frames, 
            onset_env, 
            T
        )
        
        print("Smoothing motion...")
        # Smooth over time for natural movement
        pose_seq = smooth_motion(pose_seq, window=5)
        
        # Ensure skeleton stays centered and properly scaled
        # Center around origin
        center = np.mean(pose_seq, axis=1, keepdims=True)  # (T, 1, 2)
        pose_seq = pose_seq - center
        
        # Scale to fit display nicely (keep proportions)
        max_range = np.abs(pose_seq).max()
        if max_range > 0:
            pose_seq = pose_seq / max_range * 0.85  # Scale to fit [-0.85, 0.85] range

        print(f"\n✓ Successfully generated {T} frames of dance animation!")
        print(f"  Audio duration: {len(y)/sr:.2f} seconds")
        print(f"  Animation will play at ~{sr/len(y)*T:.1f} fps\n")
        
        return pose_seq, y, sr
        
    except Exception as e:
        print(f"\n❌ Error generating dance: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
