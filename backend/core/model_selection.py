import pandas as pd
import numpy as np

def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detects if the task is regression or classification based on the target column.
    """
    if target_col not in df.columns:
        return "unknown"
        
    dtype = df[target_col].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        # Could be regression or classification. Check unique values.
        if df[target_col].nunique() < 20:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"

def detect_target_leakage(df: pd.DataFrame, target_col: str) -> list:
    """
    Detects features that might be leaking the target.
    A simple heuristic is checking for extremely high correlation with numeric targets, 
    or near-perfect association with categorical targets.
    """
    leaking_features = []
    if target_col not in df.columns or df[target_col].nunique() < 2:
        return leaking_features
        
    task_type = detect_task_type(df, target_col)
    
    if task_type == "regression":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_col in numeric_cols:
            correlations = df[numeric_cols].corr()[target_col].abs()
            for col, corr_val in correlations.items():
                if col != target_col and corr_val > 0.98:
                    leaking_features.append({
                        "feature": col,
                        "reason": f"Extremely high correlation ({corr_val:.3f}) with target.",
                        "recommendation": f"Exclude '{col}' from features."
                    })
    elif task_type == "classification":
        # For classification, we can check if any categorical feature perfectly matches the target
        # or if any numeric feature separates the classes perfectly (using simple metrics, e.g. Cramér's V or just overlap)
        # As a heuristic, we check if one feature has >99% predictive power (using a quick tree or logic)
        # For speed, we just do a quick check on categorical features matching exactly.
        for col in df.columns:
            if col != target_col and df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if df[col].nunique() == df[target_col].nunique():
                    crosstab = pd.crosstab(df[col], df[target_col])
                    # If diagonal is heavily dominant
                    max_in_row = crosstab.max(axis=1)
                    row_sums = crosstab.sum(axis=1)
                    if (max_in_row / row_sums).mean() > 0.99:
                        leaking_features.append({
                            "feature": col,
                            "reason": "Feature almost perfectly predicts classes directly.",
                            "recommendation": f"Exclude '{col}' from features."
                        })
    return leaking_features

def select_candidate_models(df: pd.DataFrame, task_type: str) -> list:
    """
    Returns a list of appropriate models to benchmark based on dataset size and task type.
    """
    candidates = []
    n_samples = len(df)
    
    if task_type == "classification":
        if n_samples < 1000:
            candidates = ["Logistic Regression", "Random Forest", "SVM"]
        elif n_samples < 50000:
            candidates = ["Random Forest", "Logistic Regression", "XGBoost"]
        else:
            candidates = ["LightGBM", "XGBoost", "CatBoost"]
            
    elif task_type == "regression":
        if n_samples < 1000:
            candidates = ["Linear Regression", "Random Forest", "Ridge"]
        elif n_samples < 50000:
            candidates = ["Random Forest", "Linear Regression", "XGBoost"]
        else:
            candidates = ["LightGBM", "XGBoost", "CatBoost"]
            
    return candidates
