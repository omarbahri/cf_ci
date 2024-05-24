import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def encode_onehot(labels):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def nll_gaussian(preds, target, variance, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target, binary=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    _, preds = preds.max(-1)
    if binary:
        preds = (preds >= 1).long()
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))


def calc_auroc(pred_edges, GT_edges):
    pred_edges = 1 - pred_edges[:, :, 0]
    return roc_auc_score(
        GT_edges.cpu().detach().flatten(),
        pred_edges.cpu().detach().flatten(),  # [:, :, 1]
    )


def auroc_per_num_influenced(preds, target, total_num_influenced):
    preds = 1 - preds[:, :, 0]
    preds = preds.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    preds_per_num_influenced = defaultdict(list)
    targets_per_num_influenced = defaultdict(list)

    for idx, k in enumerate(total_num_influenced):
        preds_per_num_influenced[k].append(preds[idx])
        targets_per_num_influenced[k].append(target[idx])

    auc_per_num_influenced = np.zeros((max(preds_per_num_influenced) + 1))
    for num_influenced, elem in preds_per_num_influenced.items():
        auc_per_num_influenced[num_influenced] = roc_auc_score(
            np.vstack(targets_per_num_influenced[num_influenced]).flatten(),
            np.vstack(elem).flatten(),
        )

    return auc_per_num_influenced

def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            losses_list[loss].append(value.item())
    return losses_list


def average_listdict(listdict, num_atoms):
    average_list = [None] * num_atoms
    for k, v in listdict.items():
        average_list[k] = sum(v) / len(v)
    return average_list