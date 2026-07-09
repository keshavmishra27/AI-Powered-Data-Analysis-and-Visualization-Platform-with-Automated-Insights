import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

def train_and_evaluate(df: pd.DataFrame, target_col: str, task_type: str, candidate_models: list) -> list:
    """
    Trains candidate models and returns a list of dictionaries with metrics and feature importances.
    """
    # Quick dropna for benchmarking (assumes user either imputed or we drop the rest)
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Very simple categorical encoding
    # In a real app we'd use a proper pipeline, but for this demo we'll just factorize
    X = df_clean.drop(columns=[target_col])
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
        
    y = df_clean[target_col]
    if task_type == "classification" and y.dtype == 'object':
        y = pd.factorize(y)[0]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    for model_name in candidate_models:
        model = _instantiate_model(model_name, task_type)
        if not model:
            continue
            
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        preds = model.predict(X_test)
        inference_time = time.time() - start_time
        
        metrics = {}
        if task_type == "classification":
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["f1"] = float(f1_score(y_test, preds, average='weighted'))
            score_col = "accuracy"
        else:
            metrics["r2"] = float(r2_score(y_test, preds))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
            score_col = "r2"
            
        # Feature importance
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Convert to float to be JSON serializable
            importances = [float(i) for i in importances]
            fi_df = pd.DataFrame({"feature": X.columns, "importance": importances})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(5)
            feature_importance = dict(zip(fi_df.feature, fi_df.importance))
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            importances = [float(i) for i in importances]
            fi_df = pd.DataFrame({"feature": X.columns, "importance": importances})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(5)
            feature_importance = dict(zip(fi_df.feature, fi_df.importance))
            
        results.append({
            "model": model_name,
            "metrics": metrics,
            "train_time": round(train_time, 4),
            "inference_time": round(inference_time, 4),
            "feature_importance": feature_importance,
            "score": metrics[score_col]
        })
        
    # Sort results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def _instantiate_model(name: str, task_type: str):
    if task_type == "classification":
        if name == "Random Forest": return RandomForestClassifier(n_estimators=50, random_state=42)
        if name == "Logistic Regression": return LogisticRegression(max_iter=500, random_state=42)
        if name == "SVM": return SVC(kernel='linear', random_state=42)
        if name == "XGBoost": return xgb.XGBClassifier(n_estimators=50, random_state=42)
        if name == "LightGBM": return lgb.LGBMClassifier(n_estimators=50, random_state=42)
        if name == "CatBoost": return CatBoostClassifier(n_estimators=50, random_state=42, verbose=0)
    else:
        if name == "Random Forest": return RandomForestRegressor(n_estimators=50, random_state=42)
        if name == "Linear Regression": return LinearRegression()
        if name == "Ridge": return Ridge(random_state=42)
        if name == "XGBoost": return xgb.XGBRegressor(n_estimators=50, random_state=42)
        if name == "LightGBM": return lgb.LGBMRegressor(n_estimators=50, random_state=42)
        if name == "CatBoost": return CatBoostRegressor(n_estimators=50, random_state=42, verbose=0)
    return None
