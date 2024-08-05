from xgboost import XGBClassifier
import pandas as pd
import data_utils as d_utils
import general_utils as g_utils
import model_utils as m_utils
from tqdm import tqdm


def run_majority_voting():
    col_names = [
        "syll1_s_freq",
        "syll2_s_freq",
        "syll3_s_freq",
        "syll4_s_freq",
        "syll5_s_freq",
        "syll6_s_freq",
        "syll7_s_freq",
        "syll8_s_freq",
        "syll9_s_freq",
        "syll10_s_freq",
        "syll1_e_freq",
        "syll2_e_freq",
        "syll3_e_freq",
        "syll4_e_freq",
        "syll5_e_freq",
        "syll6_e_freq",
        "syll7_e_freq",
        "syll8_e_freq",
        "syll9_e_freq",
        "syll10_e_freq",
        "syll1_dist",
        "syll2_dist",
        "syll3_dist",
        "syll4_dist",
        "syll5_dist",
        "syll6_dist",
        "syll7_dist",
        "syll8_dist",
        "syll9_dist",
        "syll10_dist",
        "syll1_dur",
        "syll2_dur",
        "syll3_dur",
        "syll4_dur",
        "syll5_dur",
        "syll6_dur",
        "syll7_dur",
        "syll8_dur",
        "syll9_dur",
        "syll10_dur",
        "mother_gen",
        "pup_sex",
        "avg_ISI_time",
        "pup_age",
        "session",
        "pup_strain",
        "pup_gen",
        "mouse_idx",
    ]
    seed = 25
    propotion = 0.7
    k_fold = 5
    dataset = pd.read_csv(
        r"..\data\processed_data_for_final_classification_REDUCTION_BY_RECORDING_ALLDATA.csv",
        header=None,
        names=col_names,
    )
    train_validation_dataset, test_dataset, train_validation_ids, test_ids = (
        d_utils.get_train_val_test_dataset(dataset, propotion)
    )

    k_fold_results_per_estimators = []
    for n_estimators in tqdm(range(5, 200, 25)):
        model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            learning_rate=0.01,
            max_depth=7,
            objective="binary:logistic",
            booster="gbtree",
            reg_lambda=1.5,
            reg_alpha=0.05,
            min_child_weight=0.1,
            scale_pos_weight=0.8,
            colsample_bytree=0.6,
        )
        k_fold_results = m_utils.k_fold_cross_validation(
            train_validation_dataset,
            k_fold,
            model,
            target_column="pup_gen",
            id_column="mouse_idx",
            split_function=d_utils.split_dataset,
            evaluation_function=m_utils.majority_voting_evaluation,
            num_estimators=n_estimators,
        )
        k_fold_results_per_estimators.append(k_fold_results)

    fig_curve = g_utils.plot_train_val_acc_curve(k_fold_results_per_estimators)
    fig_curve.savefig(r"..\Majority Voting Outputs\training_curve.png")
    # Have to chose the best index according to the curve , default is 3
    index = 3
    fig_cm_train = g_utils.plot_confusion_matrix(
        k_fold_results_per_estimators[index].training.y_trues,
        k_fold_results_per_estimators[index].training.y_preds,
        ["Wild", "ASD"],
        title="Majority Vote Training CM",
    )
    fig_cm_eval = g_utils.plot_confusion_matrix(
        k_fold_results_per_estimators[index].validation.y_trues,
        k_fold_results_per_estimators[index].validation.y_preds,
        ["Wild", "mASD"],
        title="Majority Vote Validation CM",
    )
    fig_cm_train.savefig(r"..\Majority Voting Outputs\cm_train.png")
    fig_cm_eval.savefig(r"..\Majority Voting Outputs\cm_eval.png")

    n_estimators = k_fold_results_per_estimators[index].validation.n_estimators
    model = XGBClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        learning_rate=0.01,
        max_depth=7,
        objective="binary:logistic",
        booster="gbtree",
        reg_lambda=1.5,
        reg_alpha=0.05,
        min_child_weight=0.1,
        scale_pos_weight=0.8,
        colsample_bytree=0.6,
    )

    y_true, y_pred, stat_df = m_utils.run_train_and_test(
        model, dataset, train_validation_ids, test_ids
    )
    acc_test = m_utils.accuracy_score(y_true, y_pred)
    fig_cm_test = g_utils.plot_confusion_matrix(
        y_true, y_pred, ["Wild", "mASD"], title="Majority Vote Test CM"
    )
    fig_cm_test.savefig(r"..\Majority Voting Outputs\cm_test.png")


if __name__ == "main":
    run_majority_voting()
