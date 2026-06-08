import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
df = pd.read_csv(
    r"C:\Users\HOCINE\Downloads\DatasetTelecome.csv",
    
)
df.dropna(inplace=True)
#print(df.info())


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
df.dropna(inplace=True)
#print(
#    df.duplicated().sum()
#)

df.drop_duplicates(
    inplace=True
)
df.drop(
    "customerID",
    axis=1,
    inplace=True
)
counts = df["Churn"].value_counts()

#print("Nombre de clients restés :", counts["No"])
#print("Nombre de clients partis :", counts["Yes"])

Q1=df["TotalCharges"].quantile(0.25)
Q3=df["TotalCharges"].quantile(0.75)
IQR=Q3-Q1
Lower=Q1-IQR*1.5
Upper=Q3+IQR*1.5
outliers = df[
    (df["TotalCharges"] < Lower)
    |
    (df["TotalCharges"] > Upper)
]

#print("nombre outliers: "+str( len(outliers)) )
df=df[
    (df["TotalCharges"]>=Lower) & (df["TotalCharges"]<=Upper)]

df = pd.get_dummies(
    df,
    drop_first=True
)
"""
print(
    df.select_dtypes(
        include="object"
    ).columns
)
"""
from sklearn.model_selection import train_test_split
X = df.drop(
    "Churn_Yes",
    axis=1
)

y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)
from sklearn.linear_model import LogisticRegression

#model = LogisticRegression()
model = LogisticRegression(
    class_weight="balanced"
)
model.fit(
    X_train,
    y_train
)
y_pred = model.predict(
    X_test
)
from sklearn.metrics import *


def evaluate_model(model, X_train, X_test, y_train, y_test):

    # entrainement
    model.fit(
        X_train,
        y_train
    )

    # prédiction
    y_pred = model.predict(
        X_test
    )

    # résultats
    print("Accuracy :")
    print(
        accuracy_score(
            y_test,
            y_pred
        )
    )

    print("\nConfusion Matrix :")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report :")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return y_pred
"""""
knn = KNeighborsClassifier(
    n_neighbors=6
)


evaluate_model(
    knn,
    X_train,
    X_test,
    y_train,
    y_test
)
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# entraînement + prédiction

"""clusters = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
).fit_predict(X)

df["Cluster"] = clusters

print(
    pd.crosstab(
        df["Cluster"],
        df["Churn_Yes"]
    )
)
"""
from xgboost import XGBClassifier
"""""
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

evaluate_model(
    xgb,
    X_train,
    X_test,
    y_train,
    y_test
)
"""""

"""""
scale = (
    y_train.value_counts()[False]
    /
    y_train.value_counts()[True]
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric="logloss"
)

evaluate_model(
    xgb,
    X_train,
    X_test,
    y_train,
    y_test
)
"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

evaluate_model(
    rf,
    X_train,
    X_test,
    y_train,
    y_test
)