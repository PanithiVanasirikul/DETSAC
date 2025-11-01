import torch.utils.data
import numpy as np
from datasets.nyu_vp import nyu
from datasets.yud_plus import yud
import utils.residual_functions
from utils.random import temp_seed, gen_item_seeds

import math

def augment_line_segments(X, noise_std=1/1024):
    """
    Augment line segments by adding Gaussian noise to endpoints,
    then recompute line equation and centroid.

    Args:
        X : (n, 12) np.ndarray
            [x1, y1, w1, x2, y2, w2, a, b, c, cx, cy, cw]
        noise_std : float
            Standard deviation of Gaussian noise to apply to x, y coordinates.

    Returns:
        (n, 12) np.ndarray with updated endpoints, line equation, and centroid.
    """

    X_aug = X.copy()

    # --- unpack points ---
    p1 = X[:, 0:3]
    p2 = X[:, 3:6]

    # --- add Gaussian noise to x, y (not w) ---
    noise1 = np.random.randn(*p1[:, :2].shape) * noise_std
    noise2 = np.random.randn(*p2[:, :2].shape) * noise_std

    p1_aug = p1.copy()
    p2_aug = p2.copy()
    p1_aug[:, :2] += noise1
    p2_aug[:, :2] += noise2

    # --- recompute line equation via cross product ---
    lines_aug = np.cross(p1_aug, p2_aug)
    line_norms = np.linalg.norm(lines_aug[:, :2], axis=1, keepdims=True) + 1e-8
    lines_aug = lines_aug / line_norms

    # --- recompute centroid (midpoint) ---
    c_aug = (p1_aug + p2_aug) / 2.0
    c_aug = c_aug / (c_aug[:, 2:3] + 1e-8)  # normalize homogeneous coordinate

    # --- pack back ---
    X_aug[:, 0:3] = p1_aug
    X_aug[:, 3:6] = p2_aug
    X_aug[:, 6:9] = lines_aug
    X_aug[:, 9:12] = c_aug

    return X_aug

def augment_sample(datum,
                   flip_h=True,
                   flip_v=True,
                   rotate=True,
                   shift=True,
                   scale=True,
                   shift_range=0.2,
                   scale_range=(0.8, 1.2),
                   rot_range=(-180, 180)):
    """
    Apply geometric augmentations (flip, rotate, shift, scale)
    to a vanishing-point dataset sample.

    datum["line_segments"]: (N, 12)
        [p1x,p1y,p1z, p2x,p2y,p2z, lx,ly,lz, cx,cy,cz]
        where p1, p2 are homogeneous endpoints, l is line eq, c is centroid.
    datum["VPs"]: (K, 3) homogeneous vanishing points.

    Returns:
        augmented datum (in-place modified).
    """
    datum = {**datum}

    # datum['lines'] = augment_line_segments(datum['lines'], noise_std=1/1024)

    M = np.eye(3)

    # --- Horizontal flip ---
    if flip_h and np.random.rand() < 0.5:
        F = np.array([[-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0, 0, 1]])
        M = F @ M

    # --- Vertical flip ---
    if flip_v and np.random.rand() < 0.5:
        F = np.array([[1,  0, 0],
                      [0, -1, 0],
                      [0,  0, 1]])
        M = F @ M

    # --- Rotation ---
    if rotate:
        angle_deg = np.random.uniform(*rot_range)
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0],
                      [sin_a,  cos_a, 0],
                      [0,      0,      1]])
        M = R @ M
    
    # --- Shift ---
    if shift:
        dx = np.random.uniform(-shift_range, shift_range)
        dy = np.random.uniform(-shift_range, shift_range)
        T = np.array([[1, 0, dx],
                      [0, 1, dy],
                      [0, 0, 1]])
        M = T @ M

    # --- Scale ---
    if scale:
        s = np.random.uniform(*scale_range)
        S = np.array([[s, 0, 0],
                      [0, s, 0],
                      [0, 0, 1]])
        M = S @ M

    # Inverse transpose for line equation transformation
    Mi = np.linalg.inv(M).T

    # --- Apply transforms ---
    lines = datum["line_segments"].copy()

    # p1, p2, centroid are transformed by M
    p1 = (M @ lines[:, 0:3].T).T
    p2 = (M @ lines[:, 3:6].T).T
    c  = (M @ lines[:, 9:12].T).T

    # line equation (l) transforms by inverse-transpose
    l  = (Mi @ lines[:, 6:9].T).T

    lines[:, 0:3]  = p1
    lines[:, 3:6]  = p2
    lines[:, 6:9]  = l
    lines[:, 9:12] = c
    datum["line_segments"] = lines

    # Vanishing points
    datum["VPs"] = (M @ datum["VPs"].T).T

    return datum

def label_lines(vps, line_segments, threshold=1-np.cos(2.0*np.pi/180.0)):

    residuals = utils.residual_functions.vanishing_point(torch.from_numpy(line_segments)[None, ...], torch.from_numpy(vps)).cpu().numpy()

    min_residuals = np.min(residuals, axis=0)

    inliers = min_residuals < threshold

    labels = np.argmin(residuals, axis=0) + 1
    labels *= inliers

    return labels, residuals

def prepare_sample(sample, max_num_lines, max_num_vps, generate_labels=False, residual_probs=False, augment=False):
    if augment:
        sample = augment_sample(sample)
    
    if max_num_lines < 0:
        max_num_lines = sample['line_segments'].shape[0]
    else:
        max_num_lines = max_num_lines

    lines = np.zeros((max_num_lines, 12)).astype(np.float32)
    mask = np.zeros(max_num_lines).astype(np.float32)
    vps = np.zeros((max_num_vps, 3)).astype(np.float32)

    if augment:
        idx = np.arange(sample['line_segments'].shape[0])
        np.random.shuffle(idx)
        sample['line_segments'] = sample['line_segments'][idx]
    # np.random.shuffle(sample['line_segments'])

    num_actual_line_segments = np.minimum(sample['line_segments'].shape[0], max_num_lines)
    lines[0:num_actual_line_segments, :] = sample['line_segments'][0:num_actual_line_segments, :12].copy()
    mask[0:num_actual_line_segments] = 1
    if num_actual_line_segments < max_num_lines:
        rest = max_num_lines - num_actual_line_segments
        lines[num_actual_line_segments:num_actual_line_segments + rest, :] = lines[0:rest, :].copy()
        mask[num_actual_line_segments:num_actual_line_segments + rest] = 1

    num_actual_vps = np.minimum(sample['VPs'].shape[0], max_num_vps)
    vps[0:num_actual_vps, :] = sample['VPs'][0:num_actual_vps]

    centroids = lines[:, 9:11]
    lengths = np.linalg.norm(lines[:, 0:3] - lines[:, 3:6], axis=-1)[:, None]
    vectors = lines[:, 0:3] - lines[:, 3:6]
    angles = np.abs(np.arctan2(vectors[:, 0], vectors[:, 1]))[:, None]

    features = np.concatenate([centroids, lengths, angles], axis=-1)

    if generate_labels:
        labels, residuals = label_lines(vps, lines)
    else:
        labels = 0
    
    if residual_probs:
        beta = 5
        tau = 1-np.cos(2.0*np.pi/180.0)
        inlier_scores = 1 - (1 / (1 + np.exp(-(beta * residuals / tau - beta))))
        inlier_scores = np.transpose(inlier_scores, (-1, -2))
    else:
        inlier_scores = 0

    return features, lines, labels, vps, 0, 0, mask, inlier_scores
    # return features, lines, labels, vps, 0, 0, mask, inlier_scores, np.transpose(residuals, (1, 0))



class NYUVP(torch.utils.data.Dataset):

    def __init__(self, split, max_num_lines=512, max_num_vps=8, use_yud=False, use_yud_plus=False,
                 deeplsd_folder=None, cache=True, generate_labels=True,
                 return_residual_probs=False, augmentation=False,
                 seed=0):
        
        if use_yud:
            self.dataset = yud.YUDVP(split=split, normalize_coords=True, data_dir_path="./datasets/yud_plus/data",
                                     yudplus=use_yud_plus, keep_in_memory=cache, external_lines_folder=deeplsd_folder)
        else:
            self.dataset = nyu.NYUVP(split=split, normalise_coordinates=True, data_dir_path="./datasets/nyu_vp/data",
                                     keep_data_in_memory=cache, external_lines_folder=deeplsd_folder)
        self.max_num_lines = max_num_lines
        self.max_num_vps = max_num_vps
        # if return_residual_probs:
        #     generate_labels = True
        self.generate_labels = generate_labels
        self.return_residual_probs = return_residual_probs
        self.augmentation = augmentation

        self.seed = seed
        self.item_seeds = gen_item_seeds(len(self.dataset), self.seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, k):
        sample = {**self.dataset[k]}
        with temp_seed(self.item_seeds[k]):
            return prepare_sample(sample, self.max_num_lines, self.max_num_vps, 
                                  generate_labels=self.generate_labels,
                                  residual_probs=self.return_residual_probs,
                                  augment=self.augmentation)

    def step(self):
        self.item_seeds += 1
