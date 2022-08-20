# Train
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import config
import numpy as np
import utils

"""
Counterclaim grid search:
{
    "discourse_text": "loss_type=squared_epsilon_insensitive;penalty=elasticnet;alpha=0.1;epsilon=0.1",
    "text_with_context": "loss_type=huber;penalty=elasticnet;alpha=0.01;epsilon=0.1",
    "previous_label": "loss_type=squared_error;penalty=elasticnet;alpha=0.01;epsilon=0.1",
    "sequence_sum": "loss_type=epsilon_insensitive;penalty=l1;alpha=0.0001;epsilon=0.01",
}
The end :)
Evidence
{
    "discourse_text": "loss_type=epsilon_insensitive;penalty=elasticnet;alpha=0.001;epsilon=0.01",
    "text_with_context": "loss_type=huber;penalty=elasticnet;alpha=0.0001;epsilon=0.01",
    "previous_label": "loss_type=squared_error;penalty=elasticnet;alpha=0.001;epsilon=0.1",
    "sequence_sum": "loss_type=modified_huber;penalty=l2;alpha=0.1;epsilon=0.1",
}
"""


for category in utils.categories:
    if category != "Counterclaim":
        continue
    print("========================================================")
    print("Category", category)
    df = pd.read_csv(config.FP_PREPROCESSED_TRAIN_CSV)
    df = df[df.discourse_type == category]

    df_test = pd.read_csv(config.FP_PREPROCESSED_TEST_CSV)
    df_test = df_test[df_test.discourse_type == category]

    df_val = pd.read_csv(config.FP_PREPROCESSED_VAL_CSV)
    df_val = df_val[df_val.discourse_type == category]

    def make_tfidf_pipeline(clf):
        return Pipeline(
            [("vect", CountVectorizer()), ("tdidf", TfidfTransformer()), ("clf", clf)]
        )

    pipelines = [
        (
            "SGD 1 discourse_text",
            make_tfidf_pipeline(
                SGDClassifier(
                    loss="squared_epsilon_insensitive",
                    penalty="elasticnet",
                    alpha=0.1,
                    epsilon=0.1,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                )
            ),
            "discourse_text",
        ),
        (
            "SGD 1 text_with_context",
            make_tfidf_pipeline(
                SGDClassifier(
                    loss="huber",
                    penalty="elasticnet",
                    alpha=0.01,
                    epsilon=0.1,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                )
            ),
            "text_with_context",
        ),
        (
            "SGD 1 previous_label",
            SGDClassifier(
                loss="squared_error",
                penalty="elasticnet",
                alpha=0.01,
                epsilon=0.1,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            ),
            "previous_label",
        ),
        (
            "SGD 1 sequence_sum",
            SGDClassifier(
                loss="epsilon_insensitive",
                penalty="l1",
                alpha=0.0001,
                epsilon=0.01,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            ),
            "sequence_sum",
        ),
    ]

    minimum_loss_classifier = {}
    minimum_loss = {}

    feature_dfs = {"train": {}, "test": {}, "validation": {}}

    y_train = df.discourse_effectiveness
    y_test = df_test.discourse_effectiveness
    y_val = df_val.discourse_effectiveness

    trained_pipelines = {}
    for pipeline_name, _, feature_column in pipelines:
        minimum_loss_classifier[pipeline_name] = {feature_column: 100.0}
        minimum_loss[pipeline_name] = 100.00
        feature_dfs["train"][feature_column] = df[feature_column]
        feature_dfs["test"][feature_column] = df_test[feature_column]
        feature_dfs["validation"][feature_column] = df_val[feature_column]

    min_loss = 100.00

    print("Starting training")
    for pipeline_name, pipeline, feature_column in pipelines:
        X_train = feature_dfs["train"][feature_column]
        X_test = feature_dfs["test"][feature_column]
        if feature_column in ["previous_label", "sequence_sum"]:
            X_train = X_train.values.reshape(-1, 1)
            X_test = X_test.values.reshape(-1, 1)

        print("Training pipeline", pipeline_name)
        pipeline.fit(X_train, y_train)
        calibrated_clf = CalibratedClassifierCV(base_estimator=pipeline, cv="prefit")
        calibrated_clf.fit(X_train, y_train)
        proba = calibrated_clf.predict_proba(X_test)
        loss = log_loss(y_test, proba, labels=pipeline.classes_)
        print("Loss: ", loss)
        if loss < minimum_loss[pipeline_name]:
            print("Found best loss so far: ", pipeline_name, loss)
            minimum_loss[pipeline_name] = loss
            minimum_loss_classifier[pipeline_name][feature_column] = calibrated_clf

        print("=====================================================================")
        print(f"Training results ({pipeline_name}): {minimum_loss}")

    print("Ensemble training")

    probs = {}
    to_stack = []
    for key, item in minimum_loss_classifier.items():
        keys = [key for key in item.keys()]
        feature_column = keys[0]
        classifier = item[feature_column]
        X_train = feature_dfs["train"][feature_column]
        if feature_column in ["previous_label", "sequence_sum"]:
            X_train = X_train.values.reshape(-1, 1)
        clf_probs = classifier.predict_proba(X_train)
        to_stack.append(clf_probs)
        probs[key] = clf_probs

    stacked_train = np.column_stack(to_stack)
    print(stacked_train.shape)

    best_ensemble = None
    best_ensemble_loss = 100
    X_train_new = stacked_train
    ensemble = SGDClassifier(
        loss="perceptron",
        penalty="l1",
        max_iter=1000,
        random_state=42,
        alpha=0.001,
        epsilon=0.1,
    )
    ensemble.fit(X_train_new, y_train)
    ensemble = CalibratedClassifierCV(base_estimator=ensemble, cv="prefit")
    ensemble.fit(X_train_new, y_train)

    probs = {}
    to_stack = []
    for key, item in minimum_loss_classifier.items():
        keys = [key for key in item.keys()]
        feature_column = keys[0]
        classifier = item[feature_column]
        X_test = feature_dfs["test"][feature_column]
        if feature_column in ["previous_label", "sequence_sum"]:
            X_test = X_test.values.reshape(-1, 1)

        clf_probs = classifier.predict_proba(X_test)
        to_stack.append(clf_probs)
        probs[key] = clf_probs

    stacked_test = np.column_stack(to_stack)
    print("Stacked shape", stacked_test.shape)

    proba_test = ensemble.predict_proba(stacked_test)

    print("Proba shape", proba_test.shape)
    loss = log_loss(y_test, proba_test, labels=pipeline.classes_)
    print("Ensemble Loss: ", loss)
    if loss < best_ensemble_loss:
        best_ensemble_loss = loss
        best_ensemble = ensemble

    print("Best ensemble", best_ensemble)
    print("Test Loss: ", best_ensemble_loss)
    score = ensemble.score(stacked_test, y_test)
    print("Test score (accuracy): ", score)

    probs = {}
    to_stack = []
    for key, item in minimum_loss_classifier.items():
        keys = [key for key in item.keys()]
        feature_column = keys[0]
        classifier = item[feature_column]
        X_val = df_val[feature_column]

        if feature_column in ["previous_label", "sequence_sum"]:
            X_val = X_val.values.reshape(-1, 1)

        clf_probs = classifier.predict_proba(X_val)
        to_stack.append(clf_probs)
        probs[key] = clf_probs

    stacked_val = np.column_stack(to_stack)
    X_val_new = stacked_val
    proba_val = best_ensemble.predict_proba(stacked_val)
    val_loss = log_loss(y_val, proba_val, labels=pipeline.classes_)
    print("Validation loss: ", val_loss)
    score = ensemble.score(stacked_val, y_val)
    print("Validation score (accuracy): ", score)
    print("End of category ", category)

    print("========================================================")
    print("========================================================")
    print("========================================================")
