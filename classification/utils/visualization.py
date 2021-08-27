import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

CMAP_LIGHT = ListedColormap(['lightyellow', 'cyan'])
CMAP_BOLD = ['orange', 'darkblue']


def visualize_classification_2d(x, y, model, grid_step=None, feature_names=None):
    if grid_step is None:
        grid_step = 0.02
    if feature_names is None:
        feature_names = ["feature_#0", "feature_#1"]

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, cmap=CMAP_LIGHT)

    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y,
                    palette=CMAP_BOLD, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Binary classification. Model: {model}")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()
