import numpy as np


def calc_roc_curve(cls: int, positive: int, negative: int, confidence: list[tuple[int, np.ndarray]]) -> (
        list[float], list[float], list[float]):
    target_len = len(confidence)
    fpr: list[float] = [0.0] * (target_len + 1)
    tpr: list[float] = [0.0] * (target_len + 1)
    auc: float = 0
    confidence: list[tuple[int, np.ndarray]] = sorted(confidence, key=lambda it: -it[1][cls])
    for i in range(target_len):
        if cls != confidence[i][0]:
            fpr[i + 1] = fpr[i] + 1 / negative
            tpr[i + 1] = tpr[i]
            auc += tpr[i + 1] / negative
        else:
            fpr[i + 1] = fpr[i]
            tpr[i + 1] = tpr[i] + 1 / positive
    return fpr, tpr, auc


def calc_metrics(conf_matrix: np.ndarray) -> (
        list[float], list[float], list[float], list[int], list[int], float):
    cls_count = conf_matrix.shape[0]
    cls_counter = [0] * cls_count
    not_cls_counter = [0] * cls_count
    precision_list = [0.0] * cls_count
    recall_list = [0.0] * cls_count
    f1_list = [0.0] * cls_count
    all_tp = 0.0
    all_fn = 0.0
    for cls in range(cls_count):
        tp: int = conf_matrix[cls, cls]
        fp: int = conf_matrix[cls, :].sum() - tp
        fn: int = conf_matrix[:, cls].sum() - tp
        tn: int = conf_matrix.sum() - tp - fp - fn
        cls_counter[cls] = tp + fn
        not_cls_counter[cls] = tn + fp
        precision: float = tp / (tp + fp)
        recall: float = tp / (tp + fn)
        f1_score: float = 2 * precision * recall / (precision + recall)
        precision_list[cls] = precision
        recall_list[cls] = recall
        f1_list[cls] = f1_score
        all_tp += tp
        all_fn += fn
    union_recall = all_tp / (all_tp + all_fn)

    return precision_list, recall_list, f1_list, cls_counter, not_cls_counter, union_recall
