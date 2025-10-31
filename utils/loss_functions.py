import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def hungarian_logit_matches(output_logits, gt_labels):
    """
    Performs Hungarian matching between GT clusters and predicted clusters,
    ignoring GT class 0 (outliers).

    Args:
        output_logits: (B, C, N) - model outputs (logits)
        gt_labels: (B, N) - ground-truth integer labels (0 = outlier)

    Returns:
        assignments: list of lists of (gt_cluster_label, pred_cluster_index)
    """
    B, C, N = output_logits.shape
    assignments = []

    for b in range(B):
        logits = output_logits[b]  # (C, N)
        labels = gt_labels[b]      # (N,)
        unique_labels = labels.unique()
        valid_labels = unique_labels[unique_labels != 0]  # ignore 0s
        K = len(valid_labels)

        cost_matrix = torch.zeros((K, C), device=logits.device)

        for i, k in enumerate(valid_labels):
            mask = (labels == k).float()  # (N,)
            if mask.sum() == 0:
                cost_matrix[i] = 1e6  # no valid points — large cost
                continue
            bce = F.binary_cross_entropy_with_logits(
                logits, 
                mask.unsqueeze(0).expand_as(logits),
                reduction='none'
            )  # (C, N)
            cost_matrix[i] = bce.mean(dim=1)

        # Convert to numpy for Hungarian solver
        cost = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        # Store assignments (ignoring outlier class)
        try:
            
            # old_assignment = [(valid_labels[i].item(), int(col_ind[i])) for i in row_ind]
            assignment = [(valid_labels[i].item(), j) for i,j in zip(row_ind, col_ind)]
            
            # we were extremely lucky that the i in row_ind always comes out as the perfect indices. e.g. 0, 1, 2, 3 (in the sense of it is not in range but the assignment still sorts for you, and that the indices continuity is perfect). In Smh, we can't assume this and we shoul
            # for old_pair, new_pair in zip(old_assignment, assignment):
            #     if old_pair[1] != new_pair[1]:
            #         breakpoint()
            assignments.append(assignment)
            # print(old_assignment)
            # print(assignment)
            # breakpoint()
        except:
            breakpoint()

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
    B, C, N = output_logits.shape
    total_loss = 0.0
    total_count = 0

    # probs = torch.sigmoid(output_logits)  # (B, C, N)

    for b in range(B):
        labels = gt_labels[b]  # (N,)
        for gt_class, pred_class in assignments[b]:
            # mask for points of this GT cluster
            # mask = (labels == gt_class).float()  # (N,)
            mask = residual_labels[b, :, gt_class-1]
            # if mask.sum() == 0:
            #     continue

            # pred_probs = probs[b, pred_class]  # (N,)
            pred_logits = output_logits[b, pred_class]  # (N,)

            # Binary cross entropy for that matched cluster
            # loss = F.binary_cross_entropy(pred_probs, mask, reduction="mean")
            loss = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction="mean")

            total_loss += loss
            total_count += 1

    if total_count == 0:
        return torch.tensor(0.0, device=output_logits.device, requires_grad=True)

    return total_loss / total_count

def compute_unmatched_loss(output_logits, gt_labels, assignments):
    B, C, N = output_logits.shape
    total_loss = 0.0
    total_count = 0
    
    # probs = torch.sigmoid(output_logits)
    for b in range(B):
        all_pred_classes = set([i for i in range(len(output_logits[0]))])
        all_pred_classes_w_assignment = set([e[1] for e in assignments[b]])
        all_pred_classes_wo_assignment = list(all_pred_classes.difference(all_pred_classes_w_assignment))
        
        points_with_labels = gt_labels[b] != 0
        for pred_class in all_pred_classes_wo_assignment:
            # gt_points_in_pred_classes_wo_assignment = probs[b, pred_class][points_with_labels]
            gt_points_in_pred_classes_wo_assignment = output_logits[b, pred_class][points_with_labels]
            # loss = F.binary_cross_entropy(gt_points_in_pred_classes_wo_assignment, 
            #                               torch.zeros(gt_points_in_pred_classes_wo_assignment.shape).to(output_logits.device))
            loss = F.binary_cross_entropy_with_logits(gt_points_in_pred_classes_wo_assignment, 
                                          torch.zeros(gt_points_in_pred_classes_wo_assignment.shape).to(output_logits.device))
            
            total_loss += loss
            total_count += 1

    if total_count == 0:
        return torch.tensor(0.0, device=output_logits.device, requires_grad=True)
    
    return total_loss/total_count

def compute_latent_code_loss(latent_logits, assignments):
    latent_logits = latent_logits.squeeze(-1)
    gt_label_rearranged = torch.zeros(latent_logits.shape[0], latent_logits.shape[1]).to(latent_logits.device)
    for i in range(len(assignments)):
        for e in assignments[i]:
            j = e[1]
            gt_label_rearranged[i, j] = 1
    loss = F.binary_cross_entropy_with_logits(latent_logits.view(-1), gt_label_rearranged.view(-1), reduction="mean")

    return loss