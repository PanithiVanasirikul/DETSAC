import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import time
import contextlib
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices



# @contextlib.contextmanager
# def timer(name):
#     t0 = time.time()
#     yield
#     print(f'[{name}] done in {time.time() - t0:.6f} s')




# def hungarian_logit_matches(output_logits, gt_labels):
#     """
#     Performs Hungarian matching between GT clusters and predicted clusters,
#     ignoring GT class 0 (outliers).

#     Args:
#         output_logits: (B, C, N) - model outputs (logits)
#         gt_labels: (B, N) - ground-truth integer labels (0 = outlier)

#     Returns:
#         assignments: list of lists of (gt_cluster_label, pred_cluster_index)
#     """
#     B, C, N = output_logits.shape
#     assignments = []

#     for b in range(B):
#         logits = output_logits[b]  # (C, N)
#         labels = gt_labels[b]      # (N,)
#         unique_labels = labels.unique()
#         valid_labels = unique_labels[unique_labels != 0]  # ignore 0s
#         K = len(valid_labels)

#         cost_matrix = torch.zeros((K, C), device=logits.device)

#         for i, k in enumerate(valid_labels):
#             mask = (labels == k).float()  # (N,)
#             if mask.sum() == 0:
#                 cost_matrix[i] = 1e6  # no valid points — large cost
#                 continue
#             bce = F.binary_cross_entropy_with_logits(
#                 logits, 
#                 mask.unsqueeze(0).expand_as(logits),
#                 reduction='none'
#             )  # (C, N)
#             cost_matrix[i] = bce.mean(dim=1)

#         # Convert to numpy for Hungarian solver
#         cost = cost_matrix.detach().cpu().numpy()
#         row_ind, col_ind = linear_sum_assignment(cost)

#         # Store assignments (ignoring outlier class)
#         try:
            
#             # old_assignment = [(valid_labels[i].item(), int(col_ind[i])) for i in row_ind]
#             assignment = [(valid_labels[i].item(), j) for i,j in zip(row_ind, col_ind)]
            
#             # we were extremely lucky that the i in row_ind always comes out as the perfect indices. e.g. 0, 1, 2, 3 (in the sense of it is not in range but the assignment still sorts for you, and that the indices continuity is perfect). In Smh, we can't assume this and we shoul
#             # for old_pair, new_pair in zip(old_assignment, assignment):
#             #     if old_pair[1] != new_pair[1]:
#             #         breakpoint()
#             assignments.append(assignment)
#             # print(old_assignment)
#             # print(assignment)
#             # breakpoint()
#         except:
#             breakpoint()

#     return assignments


def hungarian_logit_matches(output_logits, gt_labels):
    B, C, N = output_logits.shape

    # with timer("cost1"):
    #     assignments0 = []
    #     costs = []
    #     for b in range(B):
    #         logits = output_logits[b]  # (C, N)
    #         labels = gt_labels[b]      # (N,)
    #         unique_labels = labels.unique()
    #         valid_labels = unique_labels[unique_labels != 0]  # ignore 0s
    #         K = len(valid_labels)

    #         cost_matrix = torch.zeros((K, C), device=logits.device)

    #         for i, k in enumerate(valid_labels):
    #             mask = (labels == k).float()  # (N,)
    #             if mask.sum() == 0:
    #                 cost_matrix[i] = 1e6  # no valid points — large cost
    #                 continue
    #             bce = F.binary_cross_entropy_with_logits(
    #                 logits, 
    #                 mask.unsqueeze(0).expand_as(logits),
    #                 reduction='none'
    #             )  # (C, N)
    #             cost_matrix[i] = bce.mean(dim=1)

    #         # Convert to numpy for Hungarian solver
    #         cost = cost_matrix.detach().cpu().numpy()
    #         costs.append(cost)
    #         row_ind, col_ind = linear_sum_assignment(cost)
    #         assignment = [(valid_labels[i].item(), j) for i,j in zip(row_ind, col_ind)]
    #         assignments0.append(assignment)

    # with timer("cost2"):
    gt_labels_one_hot = (gt_labels[:, None, :] == torch.arange(1, C+1, device=gt_labels.device)[None, :, None]).to(output_logits.dtype)  # (B, C, N)
    bce = F.binary_cross_entropy_with_logits(
        output_logits[:, None, :, :].repeat(1, C, 1, 1),
        gt_labels_one_hot[:, :, None, :].repeat(1, 1, C, 1),
        reduction='none'
    ).mean(dim=-1)  # (B, C, C)
    gt_sel = (gt_labels_one_hot > 0).any(axis=-1)
    bce[~gt_sel] = bce.max(axis=-1)[0].sum(axis=-1).max() ** 2 + 1

    _assignment = batch_linear_assignment(bce)
    row_ind, col_ind = assignment_to_indices(_assignment)

    row_ind = row_ind.cpu().numpy()
    col_ind = col_ind.cpu().numpy()

    assignments = []
    for b, size in enumerate(gt_sel.sum(axis=-1).cpu().numpy()):
        pairs = np.stack((row_ind[b] + 1, col_ind[b]), axis=-1)[:size]
        assignments.append(pairs)

    # for a0, a1 in zip(assignments0, assignments):
    #     assert (np.array(a0) == np.array(a1)).all(), f"Mismatch {a0} vs {a1}"


    return assignments






def compute_matched_loss(output_logits, gt_labels, assignments, residual_labels):
    """
    Compute loss based on Hungarian assignments, ignoring outliers (class 0).

    Args:
        output_logits: (B, C, N)
        gt_labels: (B, N)
        assignments: list of list of (gt_class_label, pred_class_index)
    Returns:
        total_loss: scalar tensor (requires grad)
    """
    # B, C, N = output_logits.shape
    # total_loss = 0.0
    # total_count = 0

    # probs = torch.sigmoid(output_logits)  # (B, C, N)

    # start_time = time.time()

    # for b in range(B):
    #     labels = gt_labels[b]  # (N,)
    #     for gt_class, pred_class in assignments[b]:
    #         # mask for points of this GT cluster
    #         # mask = (labels == gt_class).float()  # (N,)
    #         mask = residual_labels[b, :, gt_class-1]
    #         # if mask.sum() == 0:
    #         #     continue

    #         # pred_probs = probs[b, pred_class]  # (N,)
    #         pred_logits = output_logits[b, pred_class]  # (N,)

    #         # Binary cross entropy for that matched cluster
    #         # loss = F.binary_cross_entropy(pred_probs, mask, reduction="mean")
    #         loss = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction="mean")

    #         total_loss += loss
    #         total_count += 1

    # dur1 = time.time() - start_time



    # start_time = time.time()

    ass_arr = torch.tensor(np.concatenate(assignments))
    ass_batch_idx = torch.tensor(np.concatenate([ [i] * len(x) for i, x in enumerate(assignments) ]))

    gt_sel = ass_arr[:, 0] - 1 + ass_batch_idx * residual_labels.shape[-1]
    residual_labels_sel = residual_labels.permute(0, 2, 1).reshape(-1, residual_labels.shape[1])[gt_sel]

    pred_sel = ass_arr[:, 1] + ass_batch_idx * output_logits.shape[1]
    pred_logits_sel = output_logits.view(-1, output_logits.shape[-1])[pred_sel]

    loss = F.binary_cross_entropy_with_logits(pred_logits_sel, residual_labels_sel, reduction='mean')

    return torch.nan_to_num(loss, nan=0.0)

    # dur2 = time.time() - start_time


    # if total_count > 0:
    #     assert (_loss / (total_loss / total_count) - 1).abs() < 1e-7, f"Loss mismatch {_loss.cpu().numpy()}, {(total_loss / total_count).cpu().numpy()}"
    # else:
    #     assert _loss.abs() < 1e-7, f"Loss mismatch {_loss.cpu().numpy()}"

    # assert dur2 / dur1 <= 0.05, f"Duration mismatch, {dur1} {dur2}"


    # if total_count == 0:
    #     return torch.tensor(0.0, device=output_logits.device, requires_grad=True)

    # return total_loss / total_count



def compute_unmatched_loss(output_logits, gt_labels, assignments):
    # B, C, N = output_logits.shape
    # total_loss = 0.0
    # total_count = 0
    
    # # probs = torch.sigmoid(output_logits)
    # for b in range(B):
    #     all_pred_classes = set([i for i in range(len(output_logits[0]))])
    #     all_pred_classes_w_assignment = set([e[1] for e in assignments[b]])
    #     all_pred_classes_wo_assignment = list(all_pred_classes.difference(all_pred_classes_w_assignment))
        
    #     points_with_labels = gt_labels[b] != 0
    #     for pred_class in all_pred_classes_wo_assignment:
    #         # gt_points_in_pred_classes_wo_assignment = probs[b, pred_class][points_with_labels]
    #         gt_points_in_pred_classes_wo_assignment = output_logits[b, pred_class][points_with_labels]
    #         # loss = F.binary_cross_entropy(gt_points_in_pred_classes_wo_assignment, 
    #         #                               torch.zeros(gt_points_in_pred_classes_wo_assignment.shape).to(output_logits.device))
    #         loss = F.binary_cross_entropy_with_logits(gt_points_in_pred_classes_wo_assignment, 
    #                                       torch.zeros(gt_points_in_pred_classes_wo_assignment.shape).to(output_logits.device))
            
    #         total_loss += loss
    #         total_count += 1


    ass_arr = torch.tensor(np.concatenate(assignments))
    ass_batch_idx = torch.tensor(np.concatenate([ [i] * len(x) for i, x in enumerate(assignments) ]))
    pred_sel = ass_arr[:, 1] + ass_batch_idx * output_logits.shape[1]
    not_assigned = torch.ones(output_logits.shape[:2], dtype=torch.bool, device=output_logits.device).reshape(-1)
    not_assigned[pred_sel] = False
    
    if not_assigned.sum() <= 0:
        return torch.tensor(0.0, device=output_logits.device, requires_grad=True)
    
    _output_logits = output_logits.reshape(-1, output_logits.shape[-1])[not_assigned]
    _has_labels = (gt_labels != 0)[:, None, :].repeat(1, output_logits.shape[1], 1).reshape(-1, output_logits.shape[-1])[not_assigned]
    _has_labels_cnt = _has_labels.sum(dim=-1, keepdim=True).repeat(1, _has_labels.shape[-1])[_has_labels]
    __output_logits = _output_logits[_has_labels]

    __loss = F.binary_cross_entropy_with_logits(__output_logits, torch.zeros_like(__output_logits), reduction="none") / _has_labels_cnt
    _loss = __loss.sum() / not_assigned.sum()

    return _loss

    # assert (_loss / (total_loss/total_count) - 1).abs() < 1e-6, f"Expected {_loss} to be close to {(total_loss/total_count)}"

    # if total_count == 0:
    #     return torch.tensor(0.0, device=output_logits.device, requires_grad=True)
    
    # return total_loss/total_count

def compute_latent_code_loss(latent_logits, assignments):
    # latent_logits = latent_logits.squeeze(-1)
    # gt_label_rearranged0 = torch.zeros(latent_logits.shape[0], latent_logits.shape[1]).to(latent_logits.device)
    # for i in range(len(assignments)):
    #     for e in assignments[i]:
    #         j = e[1]
    #         gt_label_rearranged0[i, j] = 1
    # loss0 = F.binary_cross_entropy_with_logits(latent_logits.view(-1), gt_label_rearranged0.view(-1), reduction="mean")
    
    # return loss0
    
    gt_label_rearranged = torch.zeros(latent_logits.shape[0] * latent_logits.shape[1]).to(latent_logits.device)
    ass_arr = torch.tensor(np.concatenate(assignments))
    ass_batch_idx = torch.tensor(np.concatenate([ [i] * len(x) for i, x in enumerate(assignments) ]))
    sel = ass_arr[:, 1] + ass_batch_idx * latent_logits.shape[1]
    gt_label_rearranged[sel] = 1

    loss = F.binary_cross_entropy_with_logits(latent_logits.view(-1), gt_label_rearranged.view(-1), reduction="mean")

    # assert (loss0 / loss - 1).abs() <= 0, f"Expected {loss0} to be close to {loss}"

    return loss