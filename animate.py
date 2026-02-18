import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sounddevice as sd

# Expanded 24-joint skeleton structure with enhanced hip/pelvis articulation
# Joint indices: 0=head, 1=neck, 2=upper_spine, 3=left_shoulder, 4=left_elbow, 5=left_wrist,
#                6=right_shoulder, 7=right_elbow, 8=right_wrist,
#                9=pelvis_center, 10=left_hip, 11=left_knee, 12=left_ankle,
#                13=right_hip, 14=right_knee, 15=right_ankle,
#                16=left_eye, 17=right_eye, 18=nose,
#                19=left_hip_front, 20=right_hip_front, 21=pelvis_back, 22=lower_spine, 23=mid_spine

# Skeleton connections forming a proper human figure with enhanced hip articulation
SKELETON_EDGES = [
    # Head
    (0, 1),  # head to neck
    (16, 0), (17, 0), (18, 0),  # eyes and nose to head
    # Spine/Torso
    (1, 2),  # neck to upper spine
    (2, 22),  # upper spine to mid spine
    (22, 21),  # mid spine to pelvis back
    (21, 9),  # pelvis back to pelvis center
    (9, 10),  # pelvis center to left hip
    (9, 13),  # pelvis center to right hip
    (10, 19),  # left hip to left hip front (pelvis articulation)
    (13, 20),  # right hip to right hip front (pelvis articulation)
    # Shoulders
    (2, 3),  # upper spine to left shoulder
    (2, 6),  # upper spine to right shoulder
    (3, 10),  # left shoulder to left hip (torso side)
    (6, 13),  # right shoulder to right hip (torso side)
    # Left arm
    (3, 4),  # left shoulder to left elbow
    (4, 5),  # left elbow to left wrist
    # Right arm
    (6, 7),  # right shoulder to right elbow
    (7, 8),  # right elbow to right wrist
    # Left leg
    (10, 11),  # left hip to left knee
    (11, 12),  # left knee to left ankle
    # Right leg
    (13, 14),  # right hip to right knee
    (14, 15),  # right knee to right ankle
]

JOINT_NAMES = [
    "head", "neck", "upper_spine", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "pelvis_center", "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_eye", "right_eye", "nose",
    "left_hip_front", "right_hip_front", "pelvis_back", "lower_spine", "mid_spine"
]


def animate_pose(pose_seq, audio, sr):
    """
    Animate a 2D skeleton dancing to audio with clear human-like visualization.
    pose_seq: (T, 24, 2) array of joint positions
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')  # Darker, more modern background
    ax.axis('off')
    ax.set_title('🎵 Dancing Skeleton 🎵', fontsize=24, fontweight='bold', 
                 color='#00ff88', pad=20, family='sans-serif')

    # Create lines for skeleton connections with better styling
    lines = []
    # Color scheme: gradient from head (purple) to limbs (green/cyan)
    edge_colors = []
    for i, (start, end) in enumerate(SKELETON_EDGES):
        # Head/spine: purple-blue
        if start in [0, 1, 2, 22, 21] or end in [0, 1, 2, 22, 21]:
            color = '#9d4edd'  # Purple
        # Arms: blue-green
        elif start in [3, 4, 5, 6, 7, 8] or end in [3, 4, 5, 6, 7, 8]:
            color = '#4cc9f0'  # Cyan-blue
        # Pelvis/hips: bright cyan
        elif start in [9, 10, 13, 19, 20, 21] or end in [9, 10, 13, 19, 20, 21]:
            color = '#00ff88'  # Bright green-cyan
        # Legs: yellow-green
        else:
            color = '#ffd60a'  # Yellow
        
        line, = ax.plot([], [], color=color, linewidth=5, alpha=0.95, 
                       zorder=1, solid_capstyle='round')
        lines.append(line)
    
    # Create points for joints with glow effect simulation
    # Hide pelvis_center (joint 9) - the central dot
    joint_colors = []
    joint_sizes = []
    joint_visible = []  # Track which joints to show
    for j in range(len(JOINT_NAMES)):
        if j == 9:  # pelvis_center - hide this joint
            joint_colors.append('#00ff88')  # Still set color but won't be visible
            joint_sizes.append(0)  # Size 0 = invisible
            joint_visible.append(False)
        elif j in [0, 16, 17, 18]:  # head, eyes, nose
            joint_colors.append('#ffd60a')  # Bright yellow
            joint_sizes.append(12)
            joint_visible.append(True)
        elif j in [10, 13, 19, 20, 21]:  # other pelvis and hip joints (excluding 9)
            joint_colors.append('#00ff88')  # Bright cyan-green
            joint_sizes.append(11)
            joint_visible.append(True)
        elif j in [1, 2, 22]:  # spine joints
            joint_colors.append('#9d4edd')  # Purple
            joint_sizes.append(10)
            joint_visible.append(True)
        else:
            joint_colors.append('#ff006e')  # Bright pink-red
            joint_sizes.append(9)
            joint_visible.append(True)
    
    points = []
    for j in range(len(JOINT_NAMES)):
        if joint_visible[j]:  # Only create visible joints
            point, = ax.plot([], [], 'o', color=joint_colors[j], markersize=joint_sizes[j], 
                            markeredgecolor='white', markeredgewidth=2, zorder=3,
                            alpha=0.95)
            points.append(point)
        else:
            # Create invisible placeholder to maintain indexing
            point, = ax.plot([], [], 'o', markersize=0, alpha=0)
            points.append(point)

    def update(frame):
        joints = pose_seq[frame]

        # Update skeleton lines
        for line, (i, j) in zip(lines, SKELETON_EDGES):
            if i < len(joints) and j < len(joints):
                x = [joints[i, 0], joints[j, 0]]
                y = [joints[i, 1], joints[j, 1]]
                line.set_data(x, y)

        # Update joint points (skip hidden joints)
        for j, point in enumerate(points):
            if j < len(joints) and joint_visible[j]:
                point.set_data([joints[j, 0]], [joints[j, 1]])
            elif j < len(joints) and not joint_visible[j]:
                # Keep hidden joints invisible
                point.set_data([], [])

        return lines + points

    # Play audio
    sd.play(audio, sr)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(pose_seq),
        interval=33,  # ~30 fps
        blit=True,
        repeat=False
    )

    plt.tight_layout()
    plt.show()
