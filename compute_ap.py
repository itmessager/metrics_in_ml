from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def Get_ROC(y_true, pos_prob):
    pos = y_true[y_true == 1]
    neg = y_true[y_true == 0]
    threshold = np.sort(pos_prob)[::-1]
    # 按概率大小逆序排列
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0]
    fpr_all = [0]
    tpr = 0
    fpr = 0
    x_step = 1 / float(len(neg))
    y_step = 1 / float(len(pos))
    y_sum = 0  # 用于计算AUC
    for i in range(len(threshold)):
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr
    return tpr_all, fpr_all, y_sum * x_step  # 获得总体TPR，FPR和相应的AUC


if __name__ == '__main__':
    y_score = np.loadtxt('Youtput_prediction.txt', dtype=np.float32)  # np.array([])

    y_true = np.loadtxt('Youtput_true.txt', dtype=np.float32)  # np.array([])
    # print(y_score)
    # print(y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    p_r_curve = plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show(p_r_curve)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    AUC_ROC = roc_auc_score(y_true, y_score)
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.show(roc_curve)
