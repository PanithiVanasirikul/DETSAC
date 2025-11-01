from utils import \
    options, initialisation, sampling, backward, visualisation, evaluation, residual_functions, inlier_counting, metrics, postprocessing, loss_functions
import torch
import torch.nn.functional as F
import time
import numpy as np



def prepare_labels(gt_labels, residual_labels):
    unique_labels = []
    for i in range(len(gt_labels)):
        unique_labels.append(torch.sort(torch.unique(gt_labels[i]))[0])

        gt_labels[i] = (1 * (
            gt_labels[i][:, None] == unique_labels[i][None, :]
        )).argmax(axis=-1)

        if unique_labels[i][0] > 0:
            gt_labels[i] += 1

    if len(residual_labels.shape) <= 1:
        residual_labels = (
            1.0 * (
                torch.arange(gt_labels.max())[None, None, :] == \
                (gt_labels[:, :, None] - 1)
            )
        )
    else:
        for i in range(len(gt_labels)):
            residual_labels[i, :, :gt_labels[i].max()] = \
                residual_labels[i][:, unique_labels[i][unique_labels[i] != 0] - 1]
            residual_labels[i, :, gt_labels[i].max():] = 0.0

    return gt_labels, residual_labels






if __name__ == "__main__":
    gt_labels = torch.tensor([[0, 1, 2, 1, 0, 2, 3, 0],
                              [2, 7, 6, 2, 0, 0, 10, 10],
                              [2, 7, 6, 2, 1, 1, 10, 10]])
    residual_labels = torch.zeros(3)

    new_gt_labels, new_residual_labels = prepare_labels(gt_labels, residual_labels)

    valid_gt_labels = torch.tensor([[0, 1, 2, 1, 0, 2, 3, 0],
                                    [1, 3, 2, 1, 0, 0, 4, 4],
                                    [2, 4, 3, 2, 1, 1, 5, 5]])
    assert (new_gt_labels == valid_gt_labels).all(), "gt labels not matching"

    valid_residuals1 = torch.tensor([[[0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0]],
                                     [[1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0]],
                                     [[0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1]]])

    assert (new_residual_labels == valid_residuals1).all(), "residual labels not matching"

    residualx = torch.arange(3*8, dtype=torch.float32).reshape((3,8)) + 10.0

    residual_labels2 = torch.zeros((3, 8, 5))
    valid_residuals2 = torch.zeros((3, 8, 5))

    for i in range(new_gt_labels.shape[0]):
        for j in range(new_gt_labels.shape[1]):
            if gt_labels[i, j] > 0:
                residual_labels2[i, j, gt_labels[i, j]-1] = residualx[i, j]
                valid_residuals2[i, j, valid_gt_labels[i, j]-1] = residualx[i, j]

    new_gt_labels2, new_residual_labels2 = prepare_labels(gt_labels, residual_labels2)

    assert (new_gt_labels2 == valid_gt_labels).all(), "gt labels 2 not matching"
    assert (new_residual_labels2 == valid_residuals2).all(), "residual labels 2 not matching"


    print("done")





