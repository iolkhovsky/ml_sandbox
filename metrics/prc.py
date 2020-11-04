import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc


def save_prc(rec, prec, prc_auc, path):
    plt.figure()
    plt.plot(rec, prec, color='darkorange',
             lw=2, label='PR curve (area = %0.2f)' % prc_auc)
    plt.plot([0, 1.], [0.5, 0.5], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=600, format="png")


if __name__ == "__main__":
    sample = [(0.1, 0), (0.2, 1), (0.3, 0), (0.4, 0), (0.5, 1), (0.7, 1), (0.85, 0), (0.9, 1), (0.95, 1)]
    y_score, y_true = zip(*sample)
    print("y_score: ", y_score)
    print("y_true: ", y_true)
    prc_auc = average_precision_score(y_true, y_score)
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    print("PRC AUC: ", prc_auc, auc(rec, prec))
    save_prc(rec=rec, prec=prec, prc_auc=prc_auc, path="prc_curve.png")
