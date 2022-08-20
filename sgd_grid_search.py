# Train
from random import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from config import config
import utils
import os

original_df = pd.read_csv(config.FP_PREPROCESSED_TRAIN_CSV)
augmented_dfs = {}
augmented_dataset_size = 30

for category in utils.categories:
    generated_data = pd.read_csv(
        os.path.join(config.FP_GENERATED_DIR, f"{category}{augmented_dataset_size}.csv")
    )
    generated_data["discourse_type"] = category
    generated_data = generated_data.rename(
        columns={"essay_id": "discourse_id", "discourse_text": "text"}
    )
    augmented_dfs[category] = generated_data

"""
# ("SGD", SGDClassifier(loss="perceptron", penalty="l1", max_iter=500, random_state=42)),# Counterclaim: 0.103
(
    "clf",
    RandomForestClassifier(
        max_depth=20,
        n_estimators=20,
        max_features=1,
        random_state=42,
    ),
),  # Rebuttal 0.3600

# With balanced weight


{'Lead': 0.6245413339086876, 'Position': 100, 'Claim': 100, 'Evidence': 100, 'Counterclaim': 100, 'Rebuttal': 100, 'Concluding Statement': 100}
Best parameters loss_type=epsilon_insensitive;penalty=elasticnet;alpha=0.01;epsilon=0.1

{'Lead': 100, 'Position': 0.6289575955682111, 'Claim': 100, 'Evidence': 100, 'Counterclaim': 100, 'Rebuttal': 100, 'Concluding Statement': 100}
Best parameters loss_type=modified_huber;penalty=l1;alpha=0.01;epsilon=0.1

{'Lead': 100, 'Position': 100, 'Claim': 0.7205251540301961, 'Evidence': 100, 'Counterclaim': 100, 'Rebuttal': 100, 'Concluding Statement': 100}
Best parameters loss_type=squared_epsilon_insensitive;penalty=l2;alpha=0.1;epsilon=0.1

{'Lead': 100, 'Position': 100, 'Claim': 100, 'Evidence': 0.7640457224331778, 'Counterclaim': 100, 'Rebuttal': 100, 'Concluding Statement': 100}
Best parameters loss_type=log_loss;penalty=elasticnet;alpha=0.001;epsilon=0.1

{'Lead': 100, 'Position': 100, 'Claim': 100, 'Evidence': 100, 'Counterclaim': 0.12512055001651962, 'Rebuttal': 100, 'Concluding Statement': 100}
Best parameters loss_type=log_loss;penalty=l1;alpha=0.001;epsilon=0.1

{'Lead': 100, 'Position': 100, 'Claim': 100, 'Evidence': 100, 'Counterclaim': 100, 'Rebuttal': 0.3333520129815209, 'Concluding Statement': 100}
Best parameters loss_type=squared_hinge;penalty=l1;alpha=0.001;epsilon=0.1

{'Lead': 100, 'Position': 100, 'Claim': 100, 'Evidence': 100, 'Counterclaim': 100, 'Rebuttal': 100, 'Concluding Statement': 0.7388559130132253}
Best parameters loss_type=squared_epsilon_insensitive;penalty=l1;alpha=0.01;epsilon=0.1
"""
for use_data_augmentation in [False]:
    category_dataframes = {}
    minimum_loss_classifier = {}
    minimum_loss = {}
    categories = list(original_df.discourse_type.unique())
    for category in categories:
        category_dataframes[category] = original_df[
            original_df.discourse_type == category
        ]
        minimum_loss[category] = 100
        minimum_loss_classifier[category] = None

    """
    RandomForest without SVD

    {
        Public Score on submission: 0.940 loss
        
        'Lead': 0.8191397405735943, 
        'Position': 0.771527728256977,
        'Claim': 0.8643719791261492,
        'Evidence': 0.9994699601546446,
        'Counterclaim': 0.5387615637174712,
        'Rebuttal': 0.6539992282551927,
        'Concluding Statement': 0.8097786428921896
    }

    With new feature separation
    {
        'Lead': 0.6941110891699832,
        'Position': 0.745130811783443,
        'Claim': 0.8793841434314899,
        'Evidence': 1.0056515737710097,
        'Counterclaim': 0.6768397537746516,
        'Rebuttal': 0.3094147154951862,
        'Concluding Statement': 0.8245590478553934
    }
    """

best_parameters = ""

for loss_type in [
    "hinge",
    "log_loss",
    "log",
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
                df = category_dataframes[category]
                for category, df in category_dataframes.items():
                    if category != "Concluding Statement":
                        continue
                    augmented_dataset_size = 30
                    generated_data = augmented_dfs[category]
                    # print(len(df))
                    if use_data_augmentation:
                        df = pd.concat([df, generated_data], ignore_index=True)
                    # print(len(df))
                    # print("Working with category: ", category)
                    y = df.discourse_effectiveness
                    X = df["text"]
                    n_splits = 5
                    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
                    min_loss = 100
                    # print("Starting training")
                    for train_index, test_index in kfold.split(X):
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
                        X_train = X.filter(items=train_index, axis=0)
                        y_train = y.filter(items=train_index, axis=0)
                        X_test = X.filter(items=test_index, axis=0)
                        y_test = y.filter(items=test_index, axis=0)
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
                            minimum_loss_classifier[category] = pipeline
                            best_parameters = f"loss_type={loss_type};penalty={regularization};alpha={alpha};epsilon={epsilon}"

        print("=====================================================================")
        print(f"Training results - using data augmentation: {use_data_augmentation}")
        print(minimum_loss)
        print("Best parameters", best_parameters)

    # print(minimum_loss_classifier)
