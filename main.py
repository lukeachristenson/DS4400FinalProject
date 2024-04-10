import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from webscrape import feature_data, label_data


def logistic_reg(X_train, y_train, X_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return y_pred


def LDA(X_train, y_train, X_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    return y_pred


def KNN(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=3)  # can optimize later with CV
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred


def decision_tree(X_train, y_train, X_test):
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return y_pred


def eval_metrics(model, y_test, y_pred):
    print(model+":")
    print("Accuracy:",accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))


def main():
    X = feature_data
    y = label_data

    # normalize X
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    lr_pred = logistic_reg(X_train=X_train, y_train=y_train, X_test=X_test)
    lda_pred = LDA(X_train=X_train, y_train=y_train, X_test=X_test)
    knn_pred = KNN(X_train=X_train, y_train=y_train, X_test=X_test)
    dt_pred = decision_tree(X_train=X_train, y_train=y_train, X_test=X_test)

    eval_metrics("Logistic Regression", y_test, lr_pred)
    eval_metrics("LDA", y_test, lda_pred)
    eval_metrics("KNN", y_test, knn_pred)
    eval_metrics("Decision Tree", y_test, dt_pred)


if __name__ == "__main__":
    main()
