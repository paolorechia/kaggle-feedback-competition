# Train
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import config
import utils

df_train = pd.read_csv(config.FP_PREPROCESSED_TRAIN_CSV)
df_test = pd.read_csv(config.FP_PREPROCESSED_TEST_CSV)
df_val = pd.read_csv(config.FP_PREPROCESSED_VAL_CSV)

MAX_ITERS = 10000
best_per_feature = {
    "discourse_text": None,
    "text_with_context": None,
    "previous_label": None,
    "sequence_sum": None,
}

# selected_category = "Claim"
for selected_category in utils.categories:
    print("========================================================")
    print("Category", selected_category)
    for feature_column in [
        "discourse_text",
        "text_with_context",
        "previous_label",
        "sequence_sum",
    ]:
        print("========================================================")
        print("Feature Column", feature_column)
        category_dataframes_train = {}
        category_dataframes_test = {}
        category_dataframes_val = {}

        minimum_loss_classifier = {}
        minimum_loss = {}
        categories = list(df_train.discourse_type.unique())
        for category in categories:
            category_dataframes_train[category] = df_train[
                df_train.discourse_type == category
            ]
            category_dataframes_test[category] = df_test[
                df_test.discourse_type == category
            ]
            category_dataframes_val[category] = df_val[
                df_val.discourse_type == category
            ]
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
                        cat_df_train = category_dataframes_train[category]
                        cat_df_test = category_dataframes_test[category]

                        for category, cat_df_train in category_dataframes_train.items():
                            if category != selected_category:
                                continue
                            # print(len(df))
                            # print("Working with category: ", category)
                            y_train = cat_df_train.discourse_effectiveness
                            X_train = cat_df_train[feature_column]

                            y_test = cat_df_test.discourse_effectiveness
                            X_test = cat_df_test[feature_column]

                            if feature_column in ["previous_label", "sequence_sum"]:
                                X_train = X_train.values.reshape(-1, 1)
                                X_test = X_test.values.reshape(-1, 1)
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

                            pipeline.fit(X_train, y_train)
                            calibrated_clf = CalibratedClassifierCV(
                                base_estimator=pipeline, cv="prefit"
                            )
                            calibrated_clf.fit(X_train, y_train)
                            proba = calibrated_clf.predict_proba(X_test)
                            loss = log_loss(y_test, proba, labels=pipeline.classes_)
                            # print("Loss: ", loss, loss_type)
                            if loss < minimum_loss[category]:
                                # print("Found best loss so far: ", category, loss)
                                minimum_loss[category] = loss
                                minimum_loss_classifier[category] = calibrated_clf
                                best_parameters = f"loss_type={loss_type};penalty={regularization};alpha={alpha};epsilon={epsilon}"

                # print(
                #     "====================================================================="
                # )
                # print(f"Training result")
                # print(minimum_loss)
                # print("Best parameters", best_parameters)

        # print(minimum_loss_classifier)

        for category, cat_df_train in category_dataframes_train.items():
            if category != selected_category:
                continue

            cat_df_train = category_dataframes_train[category]
            cat_df_val = category_dataframes_test[category]

            y_train = cat_df_train.discourse_effectiveness
            X_train = cat_df_train[feature_column]

            y_val = cat_df_val.discourse_effectiveness
            X_val = cat_df_val[feature_column]

            if feature_column in ["previous_label", "sequence_sum"]:
                X_train = X_train.values.reshape(-1, 1)
                X_val = X_val.values.reshape(-1, 1)

            proba_val = minimum_loss_classifier[category].predict_proba(X_val)
            loss = log_loss(y_val, proba_val, labels=pipeline.classes_)
            print("Validation loss: ", loss)

        best_per_feature[feature_column] = best_parameters
        print("End of feature column.", feature_column)

        print("========================================================")
        print("========================================================")
        print("========================================================")

    print(category, best_per_feature)
print("The end :)")
