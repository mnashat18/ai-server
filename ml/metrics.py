import numpy as np


def accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    if len(y_true) == 0:
        return 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_scores = []
    for cls in range(num_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        denom = (2 * tp) + fp + fn
        if denom == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * tp) / denom)
    return float(np.mean(f1_scores))
