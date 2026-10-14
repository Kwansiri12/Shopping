"""C05 Online Purchase Classification

เปรียบเทียบ KNN, SVM Linear, SVM RBF, Decision Tree และ Naive Bayes
เลือก Best Algorithm จาก Mean CV F1 บน Training Set
และใช้ Test Set ประเมินผลหลังจากเลือกโมเดลแล้ว
"""

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "C05_online_purchase_classification.xlsx"
DATA_SHEET = "Raw_Data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1"

TARGET = "purchased"
IDENTIFIER = "visitor_id"
NUMERIC_FEATURES = ["pages_viewed", "session_minutes", "items_viewed"]
CATEGORICAL_FEATURES = ["discount_seen", "returning_visitor", "device"]
EXPECTED_COLUMNS = [
    IDENTIFIER,
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    TARGET,
]

CATEGORY_MAPS = {
    "discount_seen": {"yes": "Yes", "no": "No"},
    "returning_visitor": {"yes": "Yes", "no": "No"},
    "device": {
        "desktop": "Desktop",
        "mobile": "Mobile",
        "tablet": "Tablet",
    },
}


def normalize_category(value, column):
    """Normalize case and surrounding spaces, while rejecting unknown values."""
    if pd.isna(value):
        return value

    normalized = str(value).strip().casefold()
    try:
        return CATEGORY_MAPS[column][normalized]
    except KeyError as error:
        raise ValueError(
            f"พบค่า {column} ที่ไม่รองรับหลัง normalization: {value!r}"
        ) from error


def make_preprocessors():
    numeric_cleaning = Pipeline(
        [("fill_missing", SimpleImputer(strategy="median"))]
    )
    categorical_cleaning = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoding",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    scaled = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("fill_missing", SimpleImputer(strategy="median")),
                        ("scaling", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("categorical", categorical_cleaning, CATEGORICAL_FEATURES),
        ]
    )
    unscaled = ColumnTransformer(
        [
            ("numeric", numeric_cleaning, NUMERIC_FEATURES),
            ("categorical", categorical_cleaning, CATEGORICAL_FEATURES),
        ]
    )
    return scaled, unscaled


def save_eda_plots(df):
    """Save the required EDA plots without displaying GUI windows."""
    numeric = df[NUMERIC_FEATURES]

    figure, axis = plt.subplots(figsize=(10, 6))
    numeric.boxplot(ax=axis)
    axis.set_title("Box Plot of Numerical Features")
    axis.set_ylabel("Value")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "boxplot_outliers.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.ravel(), NUMERIC_FEATURES):
        axis.hist(df[feature].dropna(), bins=20, color="#2f80ed")
        axis.set_title(f"Distribution of {feature}")
        axis.set_xlabel(feature)
        axis.set_ylabel("Frequency")
    axes.ravel()[-1].axis("off")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "histograms_features.png", dpi=300)
    plt.close(figure)

    correlation = numeric.corr()
    figure, axis = plt.subplots(figsize=(8, 6))
    heatmap = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(NUMERIC_FEATURES)))
    axis.set_yticks(range(len(NUMERIC_FEATURES)))
    axis.set_xticklabels(NUMERIC_FEATURES, rotation=30, ha="right")
    axis.set_yticklabels(NUMERIC_FEATURES)
    axis.set_title("Correlation Heatmap of Numerical Features")
    for row in range(len(NUMERIC_FEATURES)):
        for column in range(len(NUMERIC_FEATURES)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )
    figure.colorbar(heatmap, ax=axis, label="Correlation")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=300)
    plt.close(figure)

    target_values = sorted(df[TARGET].dropna().unique())
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.ravel(), NUMERIC_FEATURES):
        grouped_values = [
            df.loc[df[TARGET] == target_value, feature].dropna()
            for target_value in target_values
        ]
        axis.boxplot(grouped_values)
        axis.set_xticks(range(1, len(target_values) + 1))
        axis.set_xticklabels([str(value) for value in target_values])
        axis.set_title(f"{feature} by {TARGET}")
        axis.set_xlabel(TARGET)
        axis.set_ylabel(feature)
    axes.ravel()[-1].axis("off")
    figure.tight_layout()
    figure.savefig(
        ARTIFACTS_DIR / "boxplot_features_by_purchased.png",
        dpi=300,
    )
    plt.close(figure)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็น: {missing_columns}")

    print("ข้อมูล 5 แถวแรก")
    print(df[EXPECTED_COLUMNS].head())
    print("\nShape ของข้อมูล =", df.shape)
    print("\nโครงสร้างข้อมูล")
    df.info()
    print("\nMissing Values")
    print(df[EXPECTED_COLUMNS].isna().sum())
    print("\nจำนวนข้อมูลซ้ำก่อน Cleaning =", df.duplicated().sum())
    print("\nการกระจายของ Target")
    print(df[TARGET].value_counts(dropna=False))
    print(df[TARGET].value_counts(normalize=True) * 100)

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].map(
            lambda value, current_column=column: normalize_category(
                value, current_column
            )
        )

    df = df.drop_duplicates().copy()
    print("\nShape หลัง normalization และลบข้อมูลซ้ำ =", df.shape)
    print("จำนวนข้อมูลซ้ำหลัง Cleaning =", df.duplicated().sum())
    print("\nCategory หลัง normalization")
    for column in CATEGORICAL_FEATURES:
        print(column, df[column].value_counts(dropna=False).to_dict())

    outlier_summary = []
    for feature in NUMERIC_FEATURES:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        count = (
            (df[feature] < lower_bound) | (df[feature] > upper_bound)
        ).sum()
        outlier_summary.append(
            {"Feature": feature, "Outlier Count": int(count)}
        )
    print("\nผลการตรวจสอบ Outlier ด้วย IQR Method")
    print(pd.DataFrame(outlier_summary).to_string(index=False))
    print("หมายเหตุ: ตรวจสอบ outlier แต่ไม่ลบข้อมูล เพราะอาจเป็นพฤติกรรมจริง")
    save_eda_plots(df)

    df = df.drop(columns=[IDENTIFIER])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaled_preprocessor, unscaled_preprocessor = make_preprocessors()
    model_configs = {
        "KNN": (
            Pipeline(
                [
                    ("preprocessing", scaled_preprocessor),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "SVM Linear": (
            Pipeline(
                [
                    ("preprocessing", scaled_preprocessor),
                    ("model", SVC(kernel="linear")),
                ]
            ),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "SVM RBF": (
            Pipeline(
                [
                    ("preprocessing", scaled_preprocessor),
                    ("model", SVC(kernel="rbf")),
                ]
            ),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            Pipeline(
                [
                    ("preprocessing", unscaled_preprocessor),
                    (
                        "model",
                        DecisionTreeClassifier(
                            random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 3, 5, 7, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Naive Bayes": (
            Pipeline(
                [
                    ("preprocessing", unscaled_preprocessor),
                    ("model", GaussianNB()),
                ]
            ),
            {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        ),
    }

    grids = {}
    for model_name, (pipeline, parameters) in model_configs.items():
        print(f"\nกำลังค้นหา hyperparameters: {model_name}")
        grid = GridSearchCV(
            pipeline,
            parameters,
            scoring=SCORING,
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        grids[model_name] = grid

    cv_results = pd.DataFrame(
        [
            {
                "Algorithm": model_name,
                "Best CV F1": grid.best_score_,
                "CV F1 Std": grid.cv_results_["std_test_score"][
                    grid.best_index_
                ],
                "Best Parameters": json.dumps(
                    grid.best_params_, ensure_ascii=False, default=str
                ),
            }
            for model_name, grid in grids.items()
        ]
    ).sort_values("Best CV F1", ascending=False)
    cv_results = cv_results.reset_index(drop=True)
    best_model_name = cv_results.loc[0, "Algorithm"]
    best_grid = grids[best_model_name]
    best_model = best_grid.best_estimator_

    print("\nผล Cross-validation บน Training Set")
    print(cv_results.to_string(index=False))
    print("\nBest Algorithm จาก Mean CV F1 =", best_model_name)
    print("Best CV F1 =", best_grid.best_score_)
    print("Best Parameters =", best_grid.best_params_)

    artifact_names = {
        "KNN": "knn_best_model.joblib",
        "SVM Linear": "svm_linear_best_model.joblib",
        "SVM RBF": "svm_rbf_best_model.joblib",
        "Decision Tree": "decision_tree_best_model.joblib",
        "Naive Bayes": "naive_bayes_best_model.joblib",
    }
    for model_name, grid in grids.items():
        joblib.dump(
            grid.best_estimator_,
            ARTIFACTS_DIR / artifact_names[model_name],
        )
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")

    results = []
    for model_name, grid in grids.items():
        prediction = grid.best_estimator_.predict(X_test)
        results.append(
            {
                "Algorithm": model_name,
                "Best CV F1": grid.best_score_,
                "Test Accuracy": accuracy_score(y_test, prediction),
                "Test Precision": precision_score(
                    y_test, prediction, zero_division=0
                ),
                "Test Recall": recall_score(
                    y_test, prediction, zero_division=0
                ),
                "Test F1": f1_score(y_test, prediction, zero_division=0),
                "Best Parameters": json.dumps(
                    grid.best_params_, ensure_ascii=False, default=str
                ),
            }
        )
        print(f"\n{model_name}")
        print("Confusion Matrix")
        print(confusion_matrix(y_test, prediction))

    results_df = pd.DataFrame(results).sort_values(
        "Best CV F1", ascending=False
    )
    results_df.to_csv(
        ARTIFACTS_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nตารางเปรียบเทียบโมเดล")
    print(results_df.to_string(index=False))

    best_prediction = best_model.predict(X_test)
    best_test_metrics = {
        "test_accuracy": accuracy_score(y_test, best_prediction),
        "test_precision": precision_score(
            y_test, best_prediction, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, best_prediction, zero_division=0
        ),
        "test_f1": f1_score(y_test, best_prediction, zero_division=0),
    }
    print(f"\nผลประเมินสุดท้ายของ Best Model: {best_model_name}")
    for name, value in best_test_metrics.items():
        print(f"{name} = {value:.4f}")
    print(confusion_matrix(y_test, best_prediction))

    metadata = {
        "best_algorithm": best_model_name,
        "selection_metric": "mean_cv_f1",
        "cv_scoring": SCORING,
        "cv_folds": CV_FOLDS,
        "best_cv_f1": float(best_grid.best_score_),
        "best_parameters": best_grid.best_params_,
        "test_metrics": {
            name: float(value)
            for name, value in best_test_metrics.items()
        },
        "dataset_file": DATA_FILE.name,
        "dataset_sheet": DATA_SHEET,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "class_distribution_after_cleaning": {
            str(key): int(value)
            for key, value in y.value_counts().sort_index().items()
        },
        "artifact_file": "best_model.joblib",
    }
    with open(
        ARTIFACTS_DIR / "best_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4, default=str)

    print("\nบันทึก Artifacts เรียบร้อยที่:", ARTIFACTS_DIR.resolve())


if __name__ == "__main__":
    main()
