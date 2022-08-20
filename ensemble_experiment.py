# Train
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import config
import numpy as np

category = "Evidence"
df = pd.read_csv(config.FP_PREPROCESSED_TRAIN_CSV)
df = df[df.discourse_type == category]


df_test = pd.read_csv(config.FP_PREPROCESSED_TEST_CSV)
df_test = df_test[df_test.discourse_type == category]


def make_tfidf_pipeline(clf):
    return Pipeline(
        [("vect", CountVectorizer()), ("tdidf", TfidfTransformer()), ("clf", clf)]
    )


pipelines = [
    (
        "SGD 1",
        make_tfidf_pipeline(
            SGDClassifier(
                loss="modified_huber",
                penalty="l1",
                alpha=0.01,
                epsilon=0.1,
                max_iter=1000,
                random_state=42,
            )
        ),
    ),
    (
        "SGD 2",
        make_tfidf_pipeline(
            SGDClassifier(
                loss="squared_epsilon_insensitive",
                penalty="l2",
                alpha=0.1,
                epsilon=0.1,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ),
    (
        "SGD 3",
        make_tfidf_pipeline(
            SGDClassifier(
                loss="squared_error",
                penalty="elasticnet",
                alpha=0.01,
                epsilon=0.1,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ),
    # (
    #     "random_forest",
    #     make_tfidf_pipeline(
    #         RandomForestClassifier(
    #             max_depth=10,
    #             n_estimators=1,
    #             max_features=1,
    #             random_state=42,
    #         )
    #     ),
    # ),
    # (
    #     "SVC",
    #     make_tfidf_pipeline(
    #         SVC(kernel="linear", C=1, probability=True, random_state=42),
    #     ),
    # ),
    # loss_type=squared_error;penalty=elasticnet;alpha=0.01;epsilon=0.1
]

minimum_loss_classifier = {}
minimum_loss = {}

trained_pipelines = {}
for pipeline_name, _ in pipelines:
    trained_pipelines[pipeline_name] = 100.00
    minimum_loss_classifier[pipeline_name] = 100.00
    minimum_loss[pipeline_name] = 100.00
min_loss = 100.00


y_train = df.discourse_effectiveness
X_train = df["text"]

y_test = df_test.discourse_effectiveness
X_test = df_test["text"]

print("Starting training")
for pipeline_name, pipeline in pipelines:
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
        minimum_loss_classifier[pipeline_name] = calibrated_clf

    print("=====================================================================")
    print(f"Training results ({pipeline_name}): {minimum_loss}")

print("Ensemble training")

probs = {}
to_stack = []
for key, classifier in minimum_loss_classifier.items():
    clf_probs = classifier.predict_proba(X_train)
    to_stack.append(clf_probs)
    probs[key] = clf_probs

stacked_train = np.column_stack(to_stack)
print(stacked_train.shape)

best_ensemble = None
best_ensemble_loss = 100
X_train_new = stacked_train
ensemble = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000)
ensemble.fit(X_train_new, y_train)

probs = {}
to_stack = []
for key, classifier in minimum_loss_classifier.items():
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

df_val = pd.read_csv(config.FP_PREPROCESSED_VAL_CSV)
df_val = df_val[df_val.discourse_type == category]
y_val = df_val.discourse_effectiveness
X_val = df_val["text"]


probs = {}
to_stack = []
for key, classifier in minimum_loss_classifier.items():
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
