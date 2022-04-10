import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


CMAP_LIGHT = ListedColormap(['lightyellow', 'cyan', 'lightsalmon', 'palegreen'])
CMAP_BOLD = ['orange', 'darkblue', 'darkred', 'green']


def visualize_2d_pca(X, components, eigen_values):
    print(f"PCA: principal components\n{components}")
    print(f"PCA: explained variance\n{eigen_values}")

    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
    for i, (comp, var) in enumerate(zip(components, eigen_values)):
        comp = comp * var  # scale component by its variance explanation power
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i}",
            linewidth=5,
            color=f"C{i + 2}",
        )
    plt.gca().set(
        aspect="equal",
        title="2-dimensional dataset with principal components",
        xlabel="first feature",
        ylabel="second feature",
    )
    plt.legend()
    plt.show()


def visualize_classification_2d(x, y, grid_step=0.005, model=None, labels_dict=None, feature_names=None, hint=""):
    if feature_names is None:
        feature_names = ["feature_#0", "feature_#1"]
    if labels_dict is None:
        labels_dict = {x: f"label_{x}" for x in set(y)}
    classes_cnt = len(set(y))

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step * (x_max - x_min)),
        np.arange(y_min, y_max, grid_step * (y_max - y_min))
    )
    
    plt.figure(figsize=(12, 9), dpi=80)
    if model is not None:
        z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=CMAP_LIGHT[:classes_cnt])

    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=[labels_dict[label] for label in y],
                    palette=CMAP_BOLD[:classes_cnt], alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    title = f"Data visualization {hint}"
    if model is not None:
        title = f"Classification visualization for model <{model}> {hint}"

    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.show()
