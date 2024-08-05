import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
import warnings
from dataclasses import dataclass, field
from typing import List, Any
import data_utils as d_u

warnings.filterwarnings("ignore")


@dataclass
class FoldResults:
    accuracies: List[float] = field(default_factory=list)
    uar_scores: List[float] = field(default_factory=list)
    y_preds: List[Any] = field(default_factory=list)
    y_preds_proba: List[Any] = field(default_factory=list)
    y_trues: List[Any] = field(default_factory=list)
    aggregated_stats_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "mouse_idx",
                "probabilities",
                "true_label",
                "samples_count",
                "fold_index",
            ]
        )
    )
    conf_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    uar_all_folds: float = 0.0
    n_estimators: int = 0
    optimized_threshold: float = 0.0


@dataclass
class KFoldResults:
    validation: FoldResults = field(default_factory=FoldResults)
    training: FoldResults = field(default_factory=FoldResults)


def majority_voting_evaluation(model, val_df, fold_index=0):
    try:
        unique_mouse_ids = val_df["mouse_idx"].unique()
        y_true = []
        y_pred = []
        stat_df = pd.DataFrame(
            columns=[
                "mouse_idx",
                "probabilities",
                "true_label",
                "samples_count",
                "fold_index",
            ]
        )
        for mouse_id in unique_mouse_ids:
            subset = val_df[val_df["mouse_idx"] == mouse_id]
            X_val = subset.iloc[:, :-2]
            true_label = subset["pup_gen"].iloc[
                0
            ]  # Assuming all samples for a mouse_id have the same label
            y_true.append(true_label)

            predictions = model.predict(X_val.values).astype(int)
            predictions_proba = model.predict_proba(X_val.values)
            new_stat_row = {
                "mouse_idx": mouse_id,
                "probabilities": np.mean(predictions_proba, axis=0)[1],
                "true_label": true_label,
                "samples_count": len(predictions_proba),
                "fold_index": fold_index,
            }
            stat_df = pd.concat(
                [stat_df, pd.DataFrame([new_stat_row])], ignore_index=True
            )

            # Calculate mean probabilities for each label
            mean_probabilities = np.mean(predictions_proba, axis=0)

            # print("Mean probability for each class:")
            # for i, mean_prob in enumerate(mean_probabilities):
            #    print(f"Class {i}: {mean_prob:.4f}")
            majority_vote = np.bincount(predictions).argmax()
            # print(str(mouse_id)+' '+ str(true_label) +' #N of predicted smaples: ' + str(len(predictions)) + ' Prob ' + str(np.mean(predictions)))
            if mean_probabilities[0] >= 0.5:
                majority_vote = 0
            else:
                majority_vote = 1
            y_pred.append(majority_vote)

        # Calculate accuracy
        correct_predictions = sum(
            [1 for true, pred in zip(y_true, y_pred) if true == pred]
        )
        accuracy = correct_predictions / len(y_true)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate Unweighted Average Recall (UAR)
        uar = recall_score(y_true, y_pred, average="macro")
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)

    return accuracy, uar, conf_matrix, y_true, y_pred, stat_df


def run_train_and_test(model, df, train_mouse_ids, val_mouse_ids):
    train_set, _, val_set = d_u.split_dataset(df, train_mouse_ids, [], val_mouse_ids)

    X_train = train_set.iloc[:, :-2]
    y_train = train_set["pup_gen"]
    model.fit(X_train, y_train)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    accuracy, uar, _, y_true, y_pred, stat_df = majority_voting_evaluation(
        model, val_set
    )
    return y_true, y_pred, stat_df


def k_fold_cross_validation(
    df: pd.DataFrame,
    k: int,
    model: Any,
    target_column: str,
    id_column: str,
    split_function: Any,
    evaluation_function: Any,
    num_estimators: int,
) -> KFoldResults:
    unique_ids = df[id_column].unique()
    random.shuffle(unique_ids)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = KFoldResults()

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_ids), start=1):
        train_ids = unique_ids[train_idx]
        val_ids = unique_ids[val_idx]
        train_set, _, val_set = split_function(df, train_ids, [], val_ids)

        X_train = train_set.iloc[:, :-2]
        y_train = train_set[target_column]
        X_val = val_set.iloc[:, :-2]
        y_val = val_set[target_column]

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate on the training set
        accuracy_train, uar_train, _, y_true_train, y_pred_train, stat_df_train = (
            evaluation_function(model, train_set, fold)
        )
        results.training.y_trues.extend(y_true_train)
        # results.training.y_preds.extend(y_pred_train)
        results.training.aggregated_stats_df = pd.concat(
            [results.training.aggregated_stats_df, stat_df_train], ignore_index=True
        )
        results.training.n_estimators = num_estimators

        # Evaluate on the validation set
        accuracy_val, uar_val, _, y_true_val, y_pred_val, stat_df_val = (
            evaluation_function(model, val_set, fold)
        )
        results.validation.y_trues.extend(y_true_val)
        # results.validation.y_preds.extend(y_pred_val)
        results.validation.aggregated_stats_df = pd.concat(
            [results.validation.aggregated_stats_df, stat_df_val], ignore_index=True
        )
        results.validation.n_estimators = num_estimators

        optimal_th = get_optimal_threshold_from_training(results.training, fold)
        results.training.optimized_threshold = optimal_th
        results.validation.optimized_threshold = optimal_th

        k_fold_train_filtered = results.training.aggregated_stats_df[
            results.training.aggregated_stats_df["fold_index"] == fold
        ]
        y_train_pred_proba = k_fold_train_filtered["probabilities"].values
        k_fold_val_filtered = results.validation.aggregated_stats_df[
            results.validation.aggregated_stats_df["fold_index"] == fold
        ]
        y_val_pred_proba = k_fold_val_filtered["probabilities"].values

        y_train_pred_optimized = (y_train_pred_proba >= 0.5).astype(int)
        y_val_pred_optimized = (y_val_pred_proba >= 0.5).astype(int)

        results.training.y_preds.extend(y_train_pred_optimized)
        results.validation.y_preds.extend(y_val_pred_optimized)

        # print(f"Fold {fold}: Validation Accuracy = {accuracy_val}, Validation UAR = {uar_val}")
        # print(f"Fold {fold}: Training Accuracy = {accuracy_train}, Training UAR = {uar_train}")

    # Calculate confusion matrix for validation

    results.validation.conf_matrix = confusion_matrix(
        results.validation.y_trues, results.validation.y_preds
    )
    results.validation.uar_all_folds = accuracy_score(
        results.validation.y_trues, results.validation.y_preds
    )

    # Calculate confusion matrix for training
    results.training.conf_matrix = confusion_matrix(
        results.training.y_trues, results.training.y_preds
    )
    results.training.uar_all_folds = accuracy_score(
        results.training.y_trues, results.training.y_preds
    )

    print(
        f"Est {num_estimators}:  Training Accuraccy = {results.training.uar_all_folds}"
    )
    print(
        f"Est {num_estimators}:  Validation Accuraccy = {results.validation.uar_all_folds}"
    )
    print("*******************")

    return results


def get_optimal_threshold_from_training(k_fold_train, fold):
    k_fold_train_filtered = k_fold_train.aggregated_stats_df[
        k_fold_train.aggregated_stats_df["fold_index"] == fold
    ]
    y_train = k_fold_train_filtered["true_label"].values
    y_train_prob = k_fold_train_filtered["probabilities"].values
    # Calculate ROC curve on the training set
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)

    # Find the optimal threshold on the training set
    optimal_idx_train = np.argmax(tpr_train - fpr_train)
    optimal_threshold_train = thresholds_train[optimal_idx_train]
    return optimal_threshold_train
