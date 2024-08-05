import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(
    y_true, y_pred, class_names, title="Confusion Matrix", cmap="Blues"
):
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = f"{c}\n{p:.1f}%"
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{c}\n{p:.1f}%"

    annot = np.array(annot)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # plt.show()
    return plt.gcf()


def plot_train_val_acc_curve(
    k_fold_results_per_estimators,
):  # Initialize lists to store the UAR values for training and validation
    uar_train_all = []
    uar_val_all = []
    iterations_all = []

    # Extract data from k_fold_results_per_estimators
    for k_fold_result in k_fold_results_per_estimators:
        uar_train = k_fold_result.training.uar_all_folds
        uar_val = k_fold_result.validation.uar_all_folds
        iterations = k_fold_result.training.n_estimators

        uar_train_all.append(uar_train)
        uar_val_all.append(uar_val)
        iterations_all.append(iterations)

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_all, uar_train_all, label="Training Accuracy", marker="o")
    plt.plot(iterations_all, uar_val_all, label="Validation Accuracy", marker="o")

    plt.xlabel("N Estimators")
    plt.ylabel("Accuracy")
    plt.title("Majority Voting Xboost Learning Curve over Iterations")
    plt.legend()
    plt.grid(True)
    # plt.show()
    return plt.gcf()
