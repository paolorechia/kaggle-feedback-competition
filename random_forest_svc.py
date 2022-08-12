# Train
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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


for use_data_augmentation in [False, True]:
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
    """

    df = category_dataframes[category]
    n = 10
    for category, df in category_dataframes.items():

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
                    # ('svd', TruncatedSVD(n_components=n)),
                    #             ('clf', RandomForestClassifier(max_depth=15, n_estimators=20, max_features=1)),
                    (
                        "clf",
                        RandomForestClassifier(
                            max_depth=2 * n,
                            n_estimators=20,
                            max_features=1,
                            random_state=42,
                        ),
                    ),
                    # (
                    #     "clf",
                    #     SVC(kernel="linear", C=0.025, probability=True, random_state=42),
                    # ),
                ]
            )
            X_train = X.filter(items=train_index, axis=0)
            y_train = y.filter(items=train_index, axis=0)
            X_test = X.filter(items=test_index, axis=0)
            y_test = y.filter(items=test_index, axis=0)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)
            loss = log_loss(y_test, proba, labels=pipeline.classes_)
            # print("Loss: ", loss)
            if loss < minimum_loss[category]:
                # print("Found best loss so far: ", category, loss)
                minimum_loss[category] = loss
                minimum_loss_classifier[category] = pipeline

    print("=====================================================================")
    print(f"Training results - using data augmentation: {use_data_augmentation}")
    print(minimum_loss)
    # print(minimum_loss_classifier)
