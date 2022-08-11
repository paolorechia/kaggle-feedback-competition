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

train_df = pd.read_csv(config.FP_PREPROCESSED_TRAIN_CSV)

category_dataframes = {}
minimum_loss_classifier = {}
minimum_loss = {}
categories = list(train_df.discourse_type.unique())
print(categories)
for category in categories:
    category_dataframes[category] = train_df[train_df.discourse_type == category]
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

# Lead SVD = 25, RandomForest, 0.73 loss
# Position SVD = 15, RandomForest, 0.63 loss
# Claim SVD = 4, RandomForest, 0.72 loss
# Evidence SVD = 6, RandomForest, 0.84 loss
# Counterclaim SVD = 18, RandomForest, 0.22 loss
# Rebuttal SVD = 3, RandomForest, 0.17 loss
# Concluding Statement SVD = 12, RandomForest, 0.69 loss

# category = "Concluding Statement"
df = category_dataframes[category]
# for n in range(2, 3):
#     print("SVD Components ", n)
n = 10
for category, df in category_dataframes.items():
    if category != "Position":
        continue
    print("Working with category: ", category)
    y = df.discourse_effectiveness
    X = df['text']
    # Uncomment to use only discourse text
    # X = df['discourse_text']
    n_splits = 5
    kfold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    min_loss = 100
    print("Starting training")
    for train_index, test_index in kfold.split(X):
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tdidf', TfidfTransformer()),
            ('svd', TruncatedSVD(n_components=n)),
#             ('clf', RandomForestClassifier(max_depth=15, n_estimators=20, max_features=1)),
            ('clf', RandomForestClassifier(max_depth=2*n, n_estimators=20, max_features=1)),
#             ('clf',  SVC(kernel="linear", C=0.025, probability=True))
        ])
        X_train = X.filter(items=train_index, axis=0)
        y_train = y.filter(items=train_index, axis=0)
        X_test = X.filter(items=test_index, axis=0)
        y_test = y.filter(items=test_index, axis=0)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)
        loss = log_loss(y_test, proba, labels=pipeline.classes_)
        print("Loss: ", loss)
        if loss < minimum_loss[category]:
            print("Found best loss so far: ", category, loss)
            minimum_loss[category] = loss
            minimum_loss_classifier[category] = pipeline


print("Training results")
print(minimum_loss)
print(minimum_loss_classifier)