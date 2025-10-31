import os.path
from torch.utils.data import Dataset
import numpy as np
import pickle
import skimage
import random

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def homography_distance(H, X1, X2):
    HX1 = (H @ X1.T).T
    HX1 /= HX1[:, [2]]

    # Transform X2 → X1 using H^-1
    H_inv = np.linalg.inv(H)
    HX2 = (H_inv @ X2.T).T
    HX2 /= HX2[:, [2]]

    # Compute symmetric transfer error
    d1 = np.sum((HX1 - X2) ** 2, axis=1)
    d2 = np.sum((HX2 - X1) ** 2, axis=1)
    distances = d1 + d2
    return distances

def augment_points_flip(points):
    if np.random.rand() < 0.5:
        points[:, 0] = -points[:, 0]
        points[:, 2] = -points[:, 2]

    if np.random.rand() < 0.5:
        points[:, 1] = -points[:, 1]
        points[:, 3] = -points[:, 3]
    
    return points

def augment_points(points,
                      rotate=True,
                      shift=True,
                      scale=False,
                      shift_range=0.2,      # ±10% of normalized range
                      scale_range=(0.9, 1.1),
                      rot_range=(-15, 15)):  # degrees
    """
    Apply random geometric augmentations to 2D points in normalized [-1, 1] range.

    Args:
        points: np.ndarray [B, 2]
        flip_h, flip_v, rotate, shift, scale: bool flags
        shift_range: float, max shift magnitude
        scale_range: (float, float), min/max scale
        rot_range: (float, float), rotation in degrees

    Returns:
        np.ndarray [B, 2] of augmented points
    """
    pts = points.copy()

    # Random rotation
    if rotate:
        angle_deg = np.random.uniform(*rot_range)
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])
        pts = pts @ R.T

    # Random shift
    if shift:
        dx = np.random.uniform(-shift_range, shift_range)
        dy = np.random.uniform(-shift_range, shift_range)
        pts += np.array([dx, dy])
    
    # Random scale
    if scale:
        s = np.random.uniform(*scale_range)
        pts *= s

        return pts, s

    # Optional: clamp to [-1, 1]
    # pts = np.clip(pts, -1, 1)

    return pts, 1

class SMH:

    def __init__(self, data_dir, split, cache_data=False, normalize_coords=True, return_images=False, shuffle=False, return_gt_models=False):

        self.data_dir = data_dir
        self.cache_data = cache_data
        self.normalize_coords = normalize_coords
        self.return_images = return_images
        self.return_gt_models = return_gt_models

        self.img_size = (1024, 1024)

        self.train_sequences = [0, 1, 2, 3, 4]
        self.val_sequences = [5]
        self.test_sequences = [6]

        self.pairs = []

        if split == "train":
            self.coarse_paths = self.train_sequences
        elif split == "val":
            self.coarse_paths = self.val_sequences
        elif split == "test":
            self.coarse_paths = self.test_sequences
        elif split == "all":
            self.coarse_paths = self.train_sequences + self.val_sequences + self.test_sequences
        else:
            assert False, "invalid split: %s" % split

        os.makedirs("./tmp/smh_pairs", exist_ok=True)
        pairs_cache_file = os.path.join("./tmp/smh_pairs", split+".pkl")
        if os.path.exists(pairs_cache_file):
            with open(pairs_cache_file, 'rb') as f:
                self.pairs = pickle.load(f)
        else:
            print("loading SMH dataset for the first time, might take a few minutes.. ")
            for coarse_path in self.coarse_paths:
                for fine_path_dir in os.scandir(os.path.join(self.data_dir, "%d" % coarse_path)):
                    if fine_path_dir.is_dir():
                        for pair_path_dir in os.scandir(fine_path_dir.path):
                            if pair_path_dir.is_dir():
                                if os.path.exists(os.path.join(pair_path_dir.path, "features_and_ground_truth.npz")):
                                    self.pairs += [pair_path_dir.path]
            self.pairs.sort()
            with open(pairs_cache_file, 'wb') as f:
                pickle.dump(self.pairs, f, pickle.HIGHEST_PROTOCOL)

        if shuffle:
            random.shuffle(self.pairs)

        print("%s dataset: %d samples" % (split, len(self.pairs)))

        self.cache_dir = None
        if cache_data:
            # cache_folders = ["/phys/ssd/tmp/smh", "/phys/ssd/slurmstorage/tmp/smh", "/tmp/smh",
            #                  "/phys/intern/tmp/smh"]
            cache_folders = ["/mnt/ssd2/se3_to_image/related_datasets/parsac_cache_folder/smh"]
            for cache_folder in cache_folders:
                try:
                    cache_folder = os.path.join(cache_folder, split)
                    os.makedirs(cache_folder, exist_ok=True)
                    self.cache_dir = cache_folder
                    print("%s is cache folder" % cache_folder)
                    break
                except:
                    print("%s unavailable" % cache_folder)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, key):

        folder = self.pairs[key]
        datum = None

        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, "%09d.pkl" % key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    datum = pickle.load(f)

        if datum is None:
            features_and_gt = np.load(os.path.join(folder, "features_and_ground_truth.npz"), allow_pickle=True)

            gt_label = features_and_gt["labels"]
            pts1 = features_and_gt["points1"][:, :2]
            pts2 = features_and_gt["points2"][:, :2]
            sideinfo = features_and_gt["ratios"]

            if self.normalize_coords:
                scale = np.max(self.img_size)

                pts1[:,0] -= self.img_size[1]/2.
                pts2[:,0] -= self.img_size[1]/2.
                pts1[:,1] -= self.img_size[0]/2.
                pts2[:,1] -= self.img_size[0]/2.
                pts1 /= (scale/2.)
                pts2 /= (scale/2.)

            datum = {'points1': pts1, 'points2': pts2, 'sideinfo': sideinfo, 'img1size': self.img_size, 'img2size': self.img_size,
                     'labels': gt_label}

            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, "%09d.pkl" % key)
                if not os.path.exists(cache_path):
                    with open(cache_path, 'wb') as f:
                        pickle.dump(datum, f, pickle.HIGHEST_PROTOCOL)


        if self.return_images:
            img1_path = os.path.join(folder, "render0.png")
            img2_path = os.path.join(folder, "render1.png")
            image1_rgb = skimage.io.imread(img1_path).astype(float)[:, :, :3]
            image2_rgb = skimage.io.imread(img2_path).astype(float)[:, :, :3]
            image1 = rgb2gray(image1_rgb)
            image2 = rgb2gray(image2_rgb)

            datum['image1'] = image1
            datum['image2'] = image2
            datum['img1'] = image1_rgb
            datum['img2'] = image2_rgb
         
        if self.return_gt_models:
            features_and_gt = np.load(os.path.join(folder, "features_and_ground_truth.npz"), allow_pickle=True)
            planes = features_and_gt['planes']
            K1 = features_and_gt['K1']
            K2 = features_and_gt['K2']
            R = features_and_gt['R']
            t = features_and_gt['t']
            planes = planes / np.linalg.norm(planes[:, :3], axis=-1, keepdims=True)
            n = planes[:, :3]
            d = planes[:, 3]
            Hs = R[None, ...] - (t[None, :, None] @ n[:, None]) / d[:, None, None]
            Hs = K2[None] @ Hs @ np.linalg.inv(K1)[None]
            datum['gt'] = Hs

        return datum


class SMHDataset(Dataset):

    def __init__(self, data_dir_path, split, max_num_points, keep_in_mem=True,
                 permute_points=True, return_images=False, return_labels=True, max_num_models=28,
                 return_gt_models=False, return_residual_probs=False, augment=False):
        if return_residual_probs:
            return_gt_models = True
        self.homdata = SMH(data_dir_path, split, cache_data=keep_in_mem, normalize_coords=True, return_images=return_images, return_gt_models=return_gt_models)
        self.max_num_points = max_num_points
        self.permute_points = permute_points
        self.return_images = return_images
        self.return_labels = return_labels
        self.max_num_models = max_num_models
        self.split = split
        
        self.return_residual_probs = return_residual_probs
        self.augment = augment

    def denormalise(self, X):
        scale = np.max(self.homdata.img_size) / 2.0
        off = (self.homdata.img_size[1] / 2.0, self.homdata.img_size[0] / 2.0)
        p1 = X[..., 0:2] * scale
        # p2 = X[..., 0:2] * scale
        p2 = X[..., 2:4] * scale
        p1[..., 0] += off[0]
        p1[..., 1] += off[1]
        p2[..., 0] += off[0]
        p2[..., 1] += off[1]

        return p1, p2

    def __len__(self):
        return len(self.homdata)

    def __getitem__(self, key):
        datum = self.homdata[key]

        if self.max_num_points <= 0:
            max_num_points = datum['points1'].shape[0]
        else:
            max_num_points = self.max_num_points

        if self.permute_points:

            perm = np.random.permutation(datum['points1'].shape[0])
            datum['points1'] = datum['points1'][perm]
            datum['points2'] = datum['points2'][perm]
            datum['sideinfo'] = datum['sideinfo'][perm]
            datum['labels'] = datum['labels'][perm]

        points = np.zeros((max_num_points, 5)).astype(np.float32)
        mask = np.zeros((max_num_points, )).astype(int)
        labels = np.zeros((max_num_points, )).astype(int)

        num_actual_points = np.minimum(datum['points1'].shape[0], max_num_points)
        points[0:num_actual_points, 0:2] = datum['points1'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 2:4] = datum['points2'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 4] = datum['sideinfo'][0:num_actual_points].copy()
        labels[0:num_actual_points] = datum['labels'][0:num_actual_points].copy()

        mask[0:num_actual_points] = 1

        if num_actual_points < max_num_points:
            for i in range(num_actual_points, max_num_points, num_actual_points):
                rest = max_num_points-i
                num = min(rest, num_actual_points)
                points[i:i+num, :] = points[0:num, :].copy()
                labels[i:i+num] = labels[0:num].copy()
                mask[i:i+num] = 1

        if self.max_num_models:
            labels[np.nonzero(labels >= self.max_num_models)] = 0

        if 'img1' in datum.keys():
            image = datum['img1']
        else:
            image = 0

        imgsize = np.array([(1024, 1024), (1024, 1024)])
        
        # Not sure if augmenting this makes sense
        # if self.augment:
        #     points[:, :4] += np.random.normal(0, 1/1024, points[:, :self.max_num_models].shape)
        
        if self.homdata.return_gt_models:
            models = np.zeros((max_num_points, 3, 3)).astype(np.float32)
            num_models = np.minimum(datum['gt'].shape[0], max_num_points)
            models[0:num_models] = datum["gt"][0:num_models].copy()
        else:
            models = 0
        
        if self.return_residual_probs:
            points1, points2 = self.denormalise(points)
            points1 = np.concatenate((points1, np.ones((points1.shape[0],1))), axis=1)
            points2 = np.concatenate((points2, np.ones((points2.shape[0],1))), axis=1)
            
            if self.augment:
                points[:, 0:2], s1 = augment_points(points[:, 0:2])
                points[:, 2:4], s2 = augment_points(points[:, 2:4])
                
                points = augment_points_flip(points)
            
            residual_labels = np.zeros((points.shape[0], self.max_num_models))
            dist_labels = np.zeros((points.shape[0], self.max_num_models))

            for i in range(len(datum['gt'])):
                H = datum['gt'][i]
                dist = homography_distance(H, points1, points2)
                dist = dist/(512**2)
                if self.augment:
                    dist = dist*s1*s2
                beta = 5
                tau = 1
                inlier_scores = 1 - (1 / (1 + np.exp(-(beta * dist / tau - beta))))
                residual_labels[:, i] = inlier_scores
                dist_labels[:, i] = dist
                residual_labels[:, i] = (labels == i+1)
        else:
            residual_labels = 0
        return points, points, labels, 0, image, imgsize, mask, residual_labels
        # return points, points, labels, 0, image, imgsize, mask, residual_labels, dist_labels



def make_vis():
    import matplotlib
    import matplotlib.pyplot as plt

    random.seed(0)

    dataset = SMH("../datasets/smh", "all", cache_data=False, normalize_coords=False, return_images=True, shuffle=True)

    target_folder = "./tmp/fig/smh"

    os.makedirs(target_folder, exist_ok=True)

    for idx in range(len(dataset)):
        print("%d / %d" % (idx+1, len(dataset)), end="\r")
        sample = dataset[idx]
        img1 = sample["img1"].astype(np.uint8)
        img2 = sample["img2"].astype(np.uint8)
        pts1 = sample["points1"]
        pts2 = sample["points2"]
        y = sample["labels"]

        num_models = np.max(y)

        cb_hex = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8e10b3", "#374009", "#aec8ea", "#56611b", "#64a8c6", "#99d8d4", "#745a50", "#46fa50", "#e09eea", "#5b2b1f", "#723f91", "#634418", "#7db0d0", "#1ae37c", "#aa462c", "#719bb7", "#463aa2", "#98f42e", "#32185b", "#364fcd", "#7e54c8", "#bb5f7f", "#d466d5", "#5a0382", "#443067", "#a76232", "#78dbc1", "#35a4b2", "#52d387", "#af5a7e", "#3ef57d", "#d6d993"]
        cb = np.array([matplotlib.colors.to_rgb(x) for x in cb_hex])

        fig = plt.figure(figsize=(4 * 4, 4 * 2), dpi=256)
        axs = fig.subplots(nrows=1, ncols=2)
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        img1g = rgb2gray(img1) * 0.5 + 128
        img2g = rgb2gray(img2) * 0.5 + 128

        axs[0].imshow(img1g, cmap='Greys_r', vmin=0, vmax=255)
        axs[1].imshow(img2g, cmap='Greys_r', vmin=0, vmax=255)

        for j, pts in enumerate([pts1, pts2]):
            ax = axs[j]

            ms = np.where(y>0, 8, 4)

            c = cb[y]

            ax.scatter(pts[:, 0], pts[:, 1], c=c, s=ms**2)

        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_vis.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(4 * 4, 4 * 2), dpi=150)
        axs = fig.subplots(nrows=1, ncols=2)
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        axs[0].imshow(img1)
        axs[1].imshow(img2)
        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_orig.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()
