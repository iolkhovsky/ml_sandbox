import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


def save_roc(fpr, tpr, roc_auc, path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1.0], [0, 1.0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=600, format="png")


if __name__ == "__main__":
    sample = [(0.1, 0), (0.2, 1), (0.3, 0), (0.4, 0), (0.5, 1), (0.7, 1), (0.85, 0), (0.9, 1), (0.95, 1)]
    y_score, y_true = zip(*sample)
    print("y_score: ", y_score)
    print("y_true: ", y_true)
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    print("ROC AUC: ", roc_auc, auc(fpr, tpr))
    save_roc(fpr=fpr, tpr=tpr, roc_auc=roc_auc, path="roc_curve.png")
