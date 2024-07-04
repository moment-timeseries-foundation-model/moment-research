### Code for ranged and VUS AUC and PRAUC adapted from https://github.com/TheDatumOrg/VUS
### Code for adjusted best F1 adapted from https://github.com/mononitogoswami/tsad-model-selection/

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from numpy.typing import NDArray


@dataclass
class AnomalyDetectionMetrics:
    adjbestf1: Union[float, np.ndarray] = None
    raucroc: Union[float, np.ndarray] = None
    raucpr: Union[float, np.ndarray] = None
    vusroc: Union[float, np.ndarray] = None
    vuspr: Union[float, np.ndarray] = None
    keoghscore: Union[float, np.ndarray] = None


def get_anomaly_detection_metrics(
    anomaly_scores: NDArray, labels: NDArray, n_splits: int = 100
) -> float:
    _adjbestf1 = adjbestf1(y_true=labels, y_scores=anomaly_scores, n_splits=n_splits)
    raucroc, raucpr, vusroc, vuspr = vus_metrics(score=anomaly_scores, labels=labels)

    return AnomalyDetectionMetrics(
        adjbestf1=_adjbestf1, raucroc=raucroc, raucpr=raucpr, vusroc=vusroc, vuspr=vuspr
    )


def adjbestf1(y_true: npt.NDArray, y_scores: npt.NDArray, n_splits: int = 100):
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    adjusted_f1 = np.zeros(thresholds.shape)

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(
            score=y_scores,
            label=(y_true > 0),
            threshold=None,
            pred=y_pred,
            calc_latency=False,
        )
        adjusted_f1[i] = f1_score(y_pred, y_true)

    best_adjusted_f1 = np.max(adjusted_f1)

    return best_adjusted_f1


def keogh_score(y_true: npt.NDArray, y_scores: npt.NDArray):
    # L is the length of anomaly
    l = estimate_sliding_window_size(y_true)


def vus_metrics(score: NDArray, labels: NDArray):
    sliding_window_size = estimate_sliding_window_size(labels)
    grader = Metricor()

    try:
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(
            labels=labels, score=score, window=sliding_window_size, plot_ROC=True
        )
    except:
        R_AUC_ROC, R_AUC_PR = np.nan, np.nan

    try:
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
            labels, score, 2 * sliding_window_size
        )
    except:
        VUS_ROC, VUS_PR = np.nan, np.nan

    return R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR


def f1_score(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def get_list_anomaly(labels):
    results = []
    start = 0
    anom = False
    for i, val in enumerate(labels):
        if val == 1:
            anom = True
        else:
            if anom:
                results.append(i - start)
                anom = False
        if not anom:
            start = i
    return results


def estimate_sliding_window_size(labels: NDArray) -> int:
    # return int(np.median(get_list_anomaly(labels)))
    # This will only work of UCR Anomaly Archive datasets
    anomaly_start = np.argmax(labels)
    anomaly_end = len(labels) - np.argmax(labels[::-1])
    anomaly_length = anomaly_end - anomaly_start
    return int(anomaly_length)
    # The VUS repository has ways to estimate the sliding window size
    # when labels are not available.


class Metricor:
    def __init__(self, a=1, probability: bool = True, bias: str = "flat"):
        self.a = a
        self.probability = probability
        self.bias = bias

    def labels_conv(self, preds):
        """return indices of predicted anomaly"""
        index = np.where(preds >= 0.5)
        return index[0]

    def labels_conv_binary(self, preds):
        """return predicted label"""
        p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        p[index[0]] = 1
        return p

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1

        for i in range(start, start + AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue += bi
            if i in p:
                MyValue += bi
        return MyValue / MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        ### Changed this is better reduced if conditions
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]

        for i in Prange:
            if not (i[1] < start or i[0] > end):
                score += 1
        if score == 0:
            return 0
        else:
            return 1 / score

    def b(self, i, length):
        bias = self.bias
        if bias == "flat":
            return 1
        elif bias == "front-end bias":
            return length - i + 1
        elif bias == "back-end bias":
            return i
        else:
            if i <= length / 2:
                return i
            else:
                return length - i + 1

    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3 * score_sigma)).astype(int)

    def range_convers_new(self, label):
        """
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        """
        L = []
        i = 0
        j = 0
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i += 1
                if i >= len(label):  # ?
                    break  # ?
            j = i + 1
            # print('j'+str(j))
            if j >= len(label):
                if j == len(label):
                    L.append((i, j - 1))

                break
            while label[j] != 0:
                j += 1
                if j >= len(label):
                    L.append((i, j - 1))
                    break
            if j >= len(label):
                break
            L.append((i, j - 1))
            i = j
        return L

    def existence_reward(self, labels, preds):
        """
        labels: list of ordered pair
        preds predicted data
        """

        score = 0
        for i in labels:
            if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
                score += 1
        return score

    def num_nonzero_segments(self, x):
        count = 0
        if x[0] > 0:
            count += 1
        for i in range(1, len(x)):
            if x[i] > 0 and x[i - 1] == 0:
                count += 1
        return count

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e - s + 1) * percentage)

            x1 = np.arange(e, min(e + l0, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (2 * l0))

            x2 = np.arange(max(s - l0, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (2 * l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred

        TP = np.sum(product)

        # recall = min(TP/P,1)
        P_new = (P + np.sum(labels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0] : (seg[1] + 1)]) > 0:
                existence += 1

        existence_ratio = existence / len(L)
        # print(existence_ratio)

        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall * existence_ratio

        FP = np.sum(pred) - TP

        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP / N_new

        Precision_RangeAUC = TP / np.sum(pred)

        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(
        self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type="window"
    ):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)

        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type == "window":
            labels = self.extend_postive_range(labels, window=window)
        else:
            labels = self.extend_postive_range_individual(labels, percentage=percentage)

        L = self.range_convers_new(labels)
        TF_list = np.zeros((252, 2))
        Precision_list = np.ones(251)
        j = 0
        for i in np.linspace(0, len(score) - 1, 250).astype(int):
            threshold = score_sorted[i]
            pred = score >= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P, L)
            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = Precision

        TF_list[j + 1] = [1, 1]

        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        AUC_range = np.dot(width, height)

        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = (Precision_list[1:] + Precision_list[:-1]) / 2
        AP_range = np.dot(width_PR, height_PR)

        if plot_ROC:
            return AUC_range, AP_range, TF_list[:, 1], TF_list[:, 0], Precision_list

        return AUC_range

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if (
                sequence_original[i][1] + window // 2
                < sequence_original[i + 1][0] - window // 2
            ):
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append(
            (
                a,
                min(
                    sequence_original[len(sequence_original) - 1][1] + window // 2,
                    len(label) - 1,
                ),
            )
        )
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0
            N_labels = 0

            for seg in l:
                N_labels += np.sum(labels[seg[0] : seg[1] + 1])

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold

                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0] : seg[1] + 1], pred[seg[0] : seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence = 0
                for seg in L:
                    if (
                        np.dot(
                            labels[seg[0] : (seg[1] + 1)], pred[seg[0] : (seg[1] + 1)]
                        )
                        > 0
                    ):
                        existence += 1

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = AUC_range

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = (Precision_list[1:] + Precision_list[:-1]) / 2

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return (
            tpr_3d,
            fpr_3d,
            prec_3d,
            window_3d,
            sum(auc_3d) / len(window_3d),
            sum(ap_3d) / len(window_3d),
        )

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0
            N_labels = 0

            for seg in l:
                N_labels += np.sum(labels[seg[0] : seg[1] + 1])

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0] : seg[1] + 1], p[j][seg[0] : seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence = 0
                for seg in L:
                    if (
                        np.dot(
                            labels[seg[0] : (seg[1] + 1)], p[j][seg[0] : (seg[1] + 1)]
                        )
                        > 0
                    ):
                        existence += 1

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = AUC_range

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = (Precision_list[1:] + Precision_list[:-1]) / 2
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return (
            tpr_3d,
            fpr_3d,
            prec_3d,
            window_3d,
            sum(auc_3d) / len(window_3d),
            sum(ap_3d) / len(window_3d),
        )


def generate_curve(label, score, slidingWindow, version="opt", thre=250):
    if version == "opt_mem":
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = (
            Metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
        )
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = (
            Metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
        )

    X = np.array(tpr_3d).reshape(1, -1).ravel()
    X_ap = np.array(tpr_3d)[:, :-1].reshape(1, -1).ravel()
    Y = np.array(fpr_3d).reshape(1, -1).ravel()
    W = np.array(prec_3d).reshape(1, -1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0]) - 1)

    return Y, Z, X, X_ap, W, Z_ap, avg_auc_3d, avg_ap_3d
