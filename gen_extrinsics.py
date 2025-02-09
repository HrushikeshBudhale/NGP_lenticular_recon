import numpy as np

def rot_z(theta: float) -> np.ndarray:
    R = np.eye(3)
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    return R

def rot_y(theta: float) -> np.ndarray:
    R = np.eye(3)
    R[0, 0] = np.cos(theta)
    R[0, 2] = np.sin(theta)
    R[2, 0] = -np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R

def rot_x(theta: float) -> np.ndarray:
    R = np.eye(3)
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(theta)
    R[2, 1] = np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R

def get_rotation_facing(v: np.ndarray) -> np.ndarray:
    # generate rotation matrix such that Z axis of R is v
    Z = v / np.linalg.norm(v)
    up = np.array([0, 0, 1])
    if np.linalg.norm(np.cross(Z, up)) == 0:
        return np.eye(3)
    X = np.cross(Z, up)
    X = X / np.linalg.norm(X)
    Y = np.cross(Z, X)
    R = np.stack([X, Y, Z], axis=1)
    return R

def get_pose(position: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    Z = (look_at - position) / np.linalg.norm(look_at - position)
    pose = np.eye(4)
    pose[:3, :3] = get_rotation_facing(Z)
    pose[:3, 3] = position
    return pose

def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    # points: (N, 3)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = pose @ points.T
    return points[:3, :].T

def gen_surrounding_poses(radius: float=3, look_at: np.ndarray=np.array([0, 0, 0]), n_frames: int=30):
    z_offset = 2
    A = 1       # amplitude of the sinusoidal height change
    omega = 0.5 # angular frequency of the sinusoidal height change
    circles = 2 # number of circles around the center
    for i in range(n_frames):
        t = 2 * np.pi * (i / n_frames) * circles
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = z_offset + (A * np.sin(omega * t))
        yield get_pose(np.array([x,y,z]), look_at)

def gen_circular_poses(circle_center: np.ndarray, look_at: np.ndarray=np.array([0, 0, 0]), radius: float=0.5, n_frames: int=30):
    planer_positions = np.zeros((n_frames, 3))
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        planer_positions[i,:] = radius * np.array([np.cos(theta), np.sin(theta), 0])
    
    rotational_transform = get_pose(position=np.zeros(3), look_at=(look_at - circle_center))
    camera_centers = apply_pose(planer_positions, rotational_transform) # (N, 3)
    camera_centers = camera_centers + circle_center
    
    for camera_center in camera_centers:
        yield get_pose(camera_center, look_at)
