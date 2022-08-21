# Train
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import config
import utils

import warnings

warnings.filterwarnings("ignore")  # setting ignore as a parameter

MAX_ITERS = 100
best_per_feature = {
    "discourse_text": None,
    "text_with_context": None,
    "previous_label": None,
    "sequence_sum": None,
}


minimum_loss_classifier = {}
minimum_loss = {}
# selected_category = "Claim"
for selected_category in ["Claim"]:
    # for selected_category in utils.categories:
    category = selected_category
    train_fp = utils.get_by_category_fp(
        config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "train", category
    )
    test_fp = utils.get_by_category_fp(
        config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "test", category
    )
    val_fp = utils.get_by_category_fp(
        config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "val", category
    )

    df_train = pd.read_csv(train_fp)
    df_train = df_train[df_train.discourse_type == category]

    df_test = pd.read_csv(test_fp)
    df_test = df_test[df_test.discourse_type == category]

    df_val = pd.read_csv(val_fp)
    df_val = df_val[df_val.discourse_type == category]

    print("========================================================")
    print("Category", selected_category)
    for feature_column in [
        "flatten_text_matrix",
        # "discourse_text",
        # "text_with_context",
        # "previous_label",
        # "sequence_sum",
    ]:
        print("========================================================")
        print("Feature Column", feature_column)
        y_train = df_train.discourse_effectiveness
        X_train = df_train[feature_column]

        y_test = df_test.discourse_effectiveness
        X_test = df_test[feature_column]

        y_val = df_val.discourse_effectiveness
        X_val = df_val[feature_column]

        if feature_column in ["previous_label", "sequence_sum"]:
            X_train = X_train.values.reshape(-1, 1)
            X_test = X_test.values.reshape(-1, 1)
            X_val = X_val.values.reshape(-1, 1)

        if feature_column in ["flatten_text_matrix"]:
            X_train = utils.parse_and_pad_flattened_arrays(X_train, f"{category}-train")
            X_test = utils.parse_and_pad_flattened_arrays(X_test, f"{category}-test")
            X_val = utils.parse_and_pad_flattened_arrays(X_val, f"{category}-val")

        minimum_loss[category] = 100
        minimum_loss_classifier[category] = None

        best_parameters = ""

        for loss_type in [
            "hinge",
            "log_loss",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]:
            for regularization in ["l1", "l2", "elasticnet"]:
                for alpha in [0.1, 0.01, 0.001, 0.0001]:
                    for epsilon in [0.1, 0.01, 0.001]:
                        print(loss_type, regularization, alpha, epsilon)
                        if feature_column in ["previous_label", "sequence_sum"]:
                            pipeline = Pipeline(
                                [
                                    (
                                        "SGD",
                                        SGDClassifier(
                                            loss=loss_type,
                                            penalty=regularization,
                                            max_iter=MAX_ITERS,
                                            random_state=42,
                                            alpha=alpha,
                                            epsilon=epsilon,
                                            class_weight="balanced",
                                        ),
                                    ),
                                ]
                            )
                        elif feature_column in ["flatten_text_matrix"]:
                            pipeline = Pipeline(
                                [
                                    (
                                        "SGD",
                                        SGDClassifier(
                                            loss=loss_type,
                                            penalty=regularization,
                                            max_iter=MAX_ITERS,
                                            random_state=42,
                                            alpha=alpha,
                                            epsilon=epsilon,
                                            class_weight="balanced",
                                        ),
                                    ),
                                ]
                            )
                        else:
                            pipeline = Pipeline(
                                [
                                    ("vect", CountVectorizer()),
                                    ("tdidf", TfidfTransformer()),
                                    (
                                        "SGD",
                                        SGDClassifier(
                                            loss=loss_type,
                                            penalty=regularization,
                                            max_iter=1000,
                                            random_state=42,
                                            alpha=alpha,
                                            epsilon=epsilon,
                                            class_weight="balanced",
                                        ),
                                    ),
                                ]
                            )
                        min_loss = 100

                        print(X_train.shape, y_train.shape)

                        pipeline.fit(X_train, y_train)
                        calibrated_clf = CalibratedClassifierCV(
                            base_estimator=pipeline, cv="prefit"
                        )
                        calibrated_clf.fit(X_train, y_train)
                        proba = calibrated_clf.predict_proba(X_test)
                        loss = log_loss(y_test, proba, labels=pipeline.classes_)
                        print("Loss: ", loss)
                        if loss < minimum_loss[category]:
                            minimum_loss[category] = loss
                            minimum_loss_classifier[category] = calibrated_clf
                            best_parameters = f"loss_type={loss_type};penalty={regularization};alpha={alpha};epsilon={epsilon}"

        proba_val = minimum_loss_classifier[category].predict_proba(X_val)
        loss = log_loss(y_val, proba_val, labels=pipeline.classes_)
        print("Validation loss: ", loss)

        best_per_feature[feature_column] = best_parameters
        print("End of feature column.", feature_column)

        print("========================================================")
        print("========================================================")
        print("========================================================")

    # print(selected_category, best_per_feature)
print("The end :)")
