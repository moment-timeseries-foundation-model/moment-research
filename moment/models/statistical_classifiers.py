import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fit_svm(features: npt.NDArray, y: npt.NDArray, MAX_SAMPLES: int = 10000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=100000, gamma="scale")
    if train_size // nb_classes < 5 or train_size < 50:
        # print(f"Training SVM with {train_size} examples and {nb_classes} classes")
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm,
            {
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                "kernel": ["rbf"],
                "degree": [3],
                "gamma": ["scale"],
                "coef0": [0],
                "shrinking": [True],
                "probability": [False],
                "tol": [0.001],
                "cache_size": [200],
                "class_weight": [None],
                "verbose": [False],
                "max_iter": [10000000],
                "decision_function_shape": ["ovr"],
                # 'random_state': [None]
            },
            cv=5,
            n_jobs=10,
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y, train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]

        grid_search.fit(features, y)
        return grid_search.best_estimator_


def fit_lr(features: npt.NDArray, y: npt.NDArray, MAX_SAMPLES: int = 100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y, train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=1000000, multi_class="ovr"),
    )
    pipe.fit(features, y)
    return pipe


def fit_knn(features: npt.NDArray, y: npt.NDArray):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    pipe.fit(features, y)
    return pipe


def fit_ridge(
    train_features: npt.NDArray,
    train_y: npt.NDArray,
    valid_features: npt.NDArray,
    valid_y: npt.NDArray,
    MAX_SAMPLES: int = 100000,
):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y, train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y, train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = (
            np.sqrt(((valid_pred - valid_y) ** 2).mean())
            + np.abs(valid_pred - valid_y).mean()
        )
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]

    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr
