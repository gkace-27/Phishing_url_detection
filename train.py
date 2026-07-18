"""
train.py
========

Phishing URL Detection using Machine Learning
----------------------------------------------

This script trains, tunes, evaluates, and compares multiple machine
learning classifiers on the PhiUSIIL Phishing URL Dataset, then saves
the best-performing model to disk.

Dataset expected at: /content/PhiUSIIL_Phishing_URL_Dataset.csv

Author : Senior Machine Learning Engineer
Purpose: Final-year college project / academic demonstration
"""

# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------
import logging
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# XGBoost is optional. The script must still run if it is not installed.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phishing_detection")

# ----------------------------------------------------------------------
# 3. GLOBAL CONSTANTS
# ----------------------------------------------------------------------
DATASET_PATH = "/content/PhiUSIIL_Phishing_URL_Dataset.csv"
TARGET_COLUMN_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MODEL_OUTPUT_PATH = "model.pkl"


# ----------------------------------------------------------------------
# 4. DATA LOADING
# ----------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the phishing URL dataset from the given CSV path.

    Parameters
    ----------
    path : str
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the given path.
    Exception
        For any other error encountered while reading the file.
    """
    try:
        logger.info("Loading dataset from: %s", path)
        dataframe = pd.read_csv(path)
        logger.info("Dataset loaded successfully.")
        return dataframe
    except FileNotFoundError as exc:
        logger.error("Dataset file not found at path: %s", path)
        raise FileNotFoundError(
            f"Could not find the dataset at {path}. "
            "Please upload 'PhiUSIIL_Phishing_URL_Dataset.csv' to /content/ "
            "in your Google Colab environment."
        ) from exc
    except Exception as exc:
        logger.error("An error occurred while loading the dataset: %s", exc)
        raise


# ----------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
def explore_dataset(dataframe: pd.DataFrame, target_column: str) -> None:
    """
    Print a set of exploratory statistics and summaries about the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to explore.
    target_column : str
        Name of the target/label column.
    """
    logger.info("----- DATASET SHAPE -----")
    print(dataframe.shape)

    logger.info("----- DATASET INFO -----")
    dataframe.info()

    logger.info("----- FIRST FIVE ROWS -----")
    print(dataframe.head())

    logger.info("----- MISSING VALUES (per column) -----")
    missing_values = dataframe.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

    logger.info("----- DUPLICATE ROWS -----")
    print(f"Number of duplicate rows: {dataframe.duplicated().sum()}")

    logger.info("----- CLASS DISTRIBUTION -----")
    if target_column in dataframe.columns:
        print(dataframe[target_column].value_counts())
    else:
        logger.warning("Target column '%s' not found for class distribution.", target_column)


# ----------------------------------------------------------------------
# 6. TARGET COLUMN DETECTION
# ----------------------------------------------------------------------
def detect_target_column(dataframe: pd.DataFrame) -> str:
    """
    Automatically detect the target/label column in the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    str
        The name of the detected target column.

    Raises
    ------
    ValueError
        If no suitable target column can be identified.
    """
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            logger.info("Target column detected: '%s'", candidate)
            return candidate

    # Fallback: assume the last column is the target if it is binary.
    last_column = dataframe.columns[-1]
    if dataframe[last_column].nunique() == 2:
        logger.warning(
            "No standard target column name found. Falling back to last "
            "column '%s' since it is binary.",
            last_column,
        )
        return last_column

    raise ValueError(
        "Unable to automatically detect the target column. "
        "Please verify the dataset structure."
    )


# ----------------------------------------------------------------------
# 7. DATA PREPROCESSING
# ----------------------------------------------------------------------
def preprocess_data(
    dataframe: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and preprocess the dataset:
      - Remove duplicate rows
      - Handle missing values
      - Encode categorical (non-numeric) features
      - Separate features (X) from target (y)
      - Validate data types

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw dataset.
    target_column : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Preprocessed feature matrix X and target vector y.
    """
    logger.info("Starting data preprocessing...")
    data = dataframe.copy()

    # --- Remove duplicate rows ---
    before_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    after_rows = data.shape[0]
    logger.info("Removed %d duplicate rows.", before_rows - after_rows)

    # --- Handle missing values ---
    missing_total = data.isnull().sum().sum()
    if missing_total > 0:
        logger.info("Found %d missing values. Handling them now...", missing_total)
        for column in data.columns:
            if data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(data[column]):
                    median_value = data[column].median()
                    data[column].fillna(median_value, inplace=True)
                else:
                    mode_value = data[column].mode(dropna=True)
                    fill_value = mode_value.iloc[0] if not mode_value.empty else "unknown"
                    data[column].fillna(fill_value, inplace=True)
    else:
        logger.info("No missing values found in the dataset.")

    # --- Separate features and target ---
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not present in dataset.")

    y = data[target_column]
    x = data.drop(columns=[target_column])

    # --- Encode target if it is non-numeric ---
    if not pd.api.types.is_numeric_dtype(y):
        logger.info("Encoding non-numeric target column using LabelEncoder.")
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=target_column)

    # --- Drop columns that are pure identifiers / free text and not useful ---
    # Columns such as raw URL, Domain, or Title strings carry very high
    # cardinality and are not directly usable as numeric ML features.
    high_cardinality_text_columns = []
    for column in x.columns:
        if x[column].dtype == object:
            unique_ratio = x[column].nunique() / max(len(x[column]), 1)
            if unique_ratio > 0.5:
                high_cardinality_text_columns.append(column)

    if high_cardinality_text_columns:
        logger.info(
            "Dropping high-cardinality text columns not suitable as raw "
            "features: %s",
            high_cardinality_text_columns,
        )
        x.drop(columns=high_cardinality_text_columns, inplace=True)

    # --- Encode remaining categorical features ---
    categorical_columns = x.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_columns:
        logger.info("Encoding categorical feature columns: %s", categorical_columns)
        for column in categorical_columns:
            encoder = LabelEncoder()
            x[column] = encoder.fit_transform(x[column].astype(str))

    # --- Validate data types: ensure everything is numeric ---
    non_numeric_columns = x.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise TypeError(
            f"The following columns are still non-numeric after encoding: "
            f"{non_numeric_columns}"
        )

    # --- Ensure boolean columns are cast to integers ---
    bool_columns = x.select_dtypes(include=["bool"]).columns.tolist()
    if bool_columns:
        x[bool_columns] = x[bool_columns].astype(int)

    logger.info("Preprocessing complete. Final feature shape: %s", x.shape)
    return x, y


# ----------------------------------------------------------------------
# 8. DATA SPLITTING
# ----------------------------------------------------------------------
def split_data(x: pd.DataFrame, y: pd.Series):
    """
    Perform a stratified train/test split (80/20).

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    tuple
        x_train, x_test, y_train, y_test
    """
    logger.info("Splitting dataset into train (80%%) and test (20%%) sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split complete. Train shape: %s | Test shape: %s",
        x_train.shape,
        x_test.shape,
    )
    return x_train, x_test, y_train, y_test


# ----------------------------------------------------------------------
# 9. FEATURE SCALING
# ----------------------------------------------------------------------
def scale_features(x_train: pd.DataFrame, x_test: pd.DataFrame):
    """
    Standardize features using StandardScaler (fit on train, applied to both).

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    x_test : pd.DataFrame
        Testing features.

    Returns
    -------
    tuple
        Scaled x_train, scaled x_test, fitted scaler.
    """
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )
    return x_train_scaled, x_test_scaled, scaler


# ----------------------------------------------------------------------
# 10. MODEL EVALUATION
# ----------------------------------------------------------------------
def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model on the test set using multiple metrics.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn compatible classifier.
    x_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    model_name : str
        Human-readable name of the model.

    Returns
    -------
    dict
        Dictionary of computed evaluation metrics.
    """
    logger.info("Evaluating model: %s", model_name)
    y_pred = model.predict(x_test)

    # ROC-AUC requires probability scores or decision function values.
    try:
        y_proba = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        try:
            y_scores = model.decision_function(x_test)
            roc_auc = roc_auc_score(y_test, y_scores)
        except Exception:
            roc_auc = np.nan
            logger.warning("Could not compute ROC-AUC for %s.", model_name)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n===== {model_name} : Confusion Matrix =====")
    print(conf_matrix)
    print(f"\n===== {model_name} : Classification Report =====")
    print(report)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    }


# ----------------------------------------------------------------------
# 11. CROSS VALIDATION
# ----------------------------------------------------------------------
def perform_cross_validation(model, x, y, model_name: str) -> float:
    """
    Perform 5-fold stratified cross-validation and return the mean accuracy.

    Parameters
    ----------
    model : estimator
        A scikit-learn compatible classifier (unfitted or fitted; a clone
        is used internally by cross_val_score).
    x : pd.DataFrame
        Full feature matrix used for cross-validation.
    y : pd.Series
        Full target vector used for cross-validation.
    model_name : str
        Human-readable model name for logging.

    Returns
    -------
    float
        Mean cross-validation accuracy score.
    """
    logger.info("Performing %d-fold cross-validation for %s...", CV_FOLDS, model_name)
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, x, y, cv=cv_strategy, scoring="accuracy", n_jobs=-1)
    mean_score = scores.mean()
    logger.info(
        "%s cross-validation accuracy: %.4f (+/- %.4f)",
        model_name,
        mean_score,
        scores.std(),
    )
    return mean_score


# ----------------------------------------------------------------------
# 12. HYPERPARAMETER TUNING
# ----------------------------------------------------------------------
def tune_random_forest(x_train, y_train) -> RandomForestClassifier:
    """
    Perform RandomizedSearchCV hyperparameter tuning for Random Forest.

    Returns
    -------
    RandomForestClassifier
        The best estimator found by the search.
    """
    logger.info("Tuning Random Forest hyperparameters with RandomizedSearchCV...")
    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    logger.info("Best Random Forest parameters: %s", search.best_params_)
    return search.best_estimator_


def tune_xgboost(x_train, y_train):
    """
    Perform RandomizedSearchCV hyperparameter tuning for XGBoost.

    Returns
    -------
    XGBClassifier or None
        The best estimator found by the search, or None if XGBoost is
        not installed.
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost is not installed. Skipping XGBoost tuning.")
        return None

    logger.info("Tuning XGBoost hyperparameters with RandomizedSearchCV...")
    param_distributions = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    logger.info("Best XGBoost parameters: %s", search.best_params_)
    return search.best_estimator_


# ----------------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ----------------------------------------------------------------------
def display_feature_importance(model, feature_names, model_name: str, top_n: int = 15) -> None:
    """
    Display the top-N feature importances for tree-based models.

    Parameters
    ----------
    model : estimator
        A fitted tree-based model exposing `feature_importances_`.
    feature_names : list
        List of feature column names.
    model_name : str
        Human-readable model name.
    top_n : int
        Number of top features to display.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("%s does not support feature importance.", model_name)
        return

    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print(f"\n===== Top {top_n} Feature Importances : {model_name} =====")
    print(importance_df.head(top_n).to_string(index=False))


# ----------------------------------------------------------------------
# 14. MAIN PIPELINE
# ----------------------------------------------------------------------
def main() -> None:
    """
    Execute the full phishing URL detection ML pipeline:
    load data, explore, preprocess, split, train, tune, evaluate,
    compare, select best model, and persist it to disk.
    """
    try:
        # ---- Load dataset ----
        dataframe = load_dataset(DATASET_PATH)

        # ---- Detect target column ----
        target_column = detect_target_column(dataframe)

        # ---- Explore dataset ----
        explore_dataset(dataframe, target_column)

        # ---- Preprocess ----
        x, y = preprocess_data(dataframe, target_column)

        # ---- Split ----
        x_train, x_test, y_train, y_test = split_data(x, y)

        # ---- Scale ----
        x_train_scaled, x_test_scaled, _scaler = scale_features(x_train, x_test)

        results = []
        trained_models = {}

        # ---- Logistic Regression ----
        logger.info("Training Logistic Regression...")
        log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        log_reg.fit(x_train_scaled, y_train)
        results.append(evaluate_model(log_reg, x_test_scaled, y_test, "Logistic Regression"))
        perform_cross_validation(log_reg, x_train_scaled, y_train, "Logistic Regression")
        trained_models["Logistic Regression"] = log_reg

        # ---- Decision Tree ----
        logger.info("Training Decision Tree...")
        decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
        decision_tree.fit(x_train, y_train)
        results.append(evaluate_model(decision_tree, x_test, y_test, "Decision Tree"))
        perform_cross_validation(decision_tree, x_train, y_train, "Decision Tree")
        trained_models["Decision Tree"] = decision_tree

        # ---- Random Forest (tuned) ----
        logger.info("Training Random Forest (with hyperparameter tuning)...")
        random_forest = tune_random_forest(x_train, y_train)
        results.append(evaluate_model(random_forest, x_test, y_test, "Random Forest"))
        perform_cross_validation(random_forest, x_train, y_train, "Random Forest")
        trained_models["Random Forest"] = random_forest
        display_feature_importance(random_forest, x.columns.tolist(), "Random Forest")

        # ---- Extra Trees ----
        logger.info("Training Extra Trees...")
        extra_trees = ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )
        extra_trees.fit(x_train, y_train)
        results.append(evaluate_model(extra_trees, x_test, y_test, "Extra Trees"))
        perform_cross_validation(extra_trees, x_train, y_train, "Extra Trees")
        trained_models["Extra Trees"] = extra_trees
        display_feature_importance(extra_trees, x.columns.tolist(), "Extra Trees")

        # ---- XGBoost (tuned, optional) ----
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost (with hyperparameter tuning)...")
            xgboost_model = tune_xgboost(x_train, y_train)
            if xgboost_model is not None:
                results.append(evaluate_model(xgboost_model, x_test, y_test, "XGBoost"))
                perform_cross_validation(xgboost_model, x_train, y_train, "XGBoost")
                trained_models["XGBoost"] = xgboost_model
                display_feature_importance(xgboost_model, x.columns.tolist(), "XGBoost")
        else:
            logger.warning("XGBoost is not installed. Skipping XGBoost training.")

        # ---- Comparison table ----
        results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
        results_df.reset_index(drop=True, inplace=True)

        print("\n===== MODEL COMPARISON TABLE (sorted by F1 Score) =====")
        print(results_df.to_string(index=False))

        # ---- Select best model ----
        best_model_name = results_df.iloc[0]["Model"]
        best_model = trained_models[best_model_name]
        logger.info(
            "Best performing model: %s (F1 Score: %.4f)",
            best_model_name,
            results_df.iloc[0]["F1 Score"],
        )

        # ---- Save best model ----
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        logger.info("Best model saved to: %s", MODEL_OUTPUT_PATH)

        print("Training Completed Successfully")

    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        raise
    except KeyError as exc:
        logger.error("Key error: %s", exc)
        raise
    except TypeError as exc:
        logger.error("Type error: %s", exc)
        raise
    except ValueError as exc:
        logger.error("Value error: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("An unexpected error occurred: %s", exc)
        raise


if __name__ == "__main__":
    main()
