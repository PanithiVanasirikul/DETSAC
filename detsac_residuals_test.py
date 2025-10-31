from utils import \
    options, initialisation, sampling, backward, visualisation, evaluation, residual_functions, inlier_counting, metrics, postprocessing, loss_functions
import torch
import torch.nn.functional as F
import time

import pickle

opt = options.get_options()

initialisation.seeds(opt)

ckpt_dir, log = initialisation.setup_logging_and_checkpointing(opt)

model, optimizer, scheduler, device = initialisation.get_model(opt)

datasets = initialisation.get_dataset(opt)

all_inlier_dist = list()
all_outlier_dist = list()

# opt.modes=['train', 'val', 'test']

for epoch in range(opt.epochs):

    print("epoch %d / %d" % (epoch + 1, opt.epochs))
    epoch_start = time.time()

    dataloaders = initialisation.get_dataloader(opt, datasets, shuffle_all=False)

    for mode in opt.modes:

        assert not (dataloaders[mode] is None), "no dataloader for %s available" % mode

        print("mode: %s" % mode)
            
        for batch_idx, (features, X, gt_labels, gt_models, image, image_size, mask, residual_labels, dist_labels) in enumerate(dataloaders[mode]):
            X = X.to(device)
            features = features.to(device)
            gt_labels = gt_labels.to(device)
            gt_models = gt_models.to(device)
            image_size = image_size.to(device)
            mask = mask.to(device)
            residual_labels = residual_labels.to(device)
            dist_labels = dist_labels.to(device)
            
            # selection_labels = gt_labels[:, :, None] == ((torch.arange(0, opt.instances).view(1, -1).expand(len(gt_labels), -1) + 1)[:, None, :]).cuda()
            # selection_labels = gt_labels[:, :, None] == ((torch.arange(0, 3).view(1, -1).expand(len(gt_labels), -1) + 1)[:, None, :]).cuda()
            selection_labels = gt_labels[:, :, None] == ((torch.arange(0, 4).view(1, -1).expand(len(gt_labels), -1) + 1)[:, None, :]).cuda()
            
            all_inlier_dist += dist_labels[selection_labels].tolist()
            all_outlier_dist += dist_labels[~selection_labels].tolist()
        
        inlier_outlier_dist = dict()
        inlier_outlier_dist['inlier_dist'] = all_inlier_dist
        inlier_outlier_dist['outlier_dist'] = all_outlier_dist
                
        with open(f'datasets_statistics/{opt.dataset}_{mode}.pkl', 'wb') as f:
            pickle.dump(inlier_outlier_dist, f)
        breakpoint()