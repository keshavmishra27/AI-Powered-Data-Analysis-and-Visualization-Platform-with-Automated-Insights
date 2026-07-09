import pandas as pd
from backend.core.model_selection import detect_task_type, detect_target_leakage, select_candidate_models
from backend.core.benchmarking import train_and_evaluate

def run_ml_engineer_agent(df: pd.DataFrame, target_col: str) -> dict:
    """
    Executes the ML Engineer Agent logic.
    1. Detects task type (regression/classification).
    2. Detects target leakage.
    3. Selects candidate models.
    4. Benchmarks models and evaluates feature importance.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}
        
    task_type = detect_task_type(df, target_col)
    
    leakage_warnings = detect_target_leakage(df, target_col)
    
    # Optional: Automatically drop leaking features or just warn. 
    # For a real Data Scientist agent, we might drop them or ask the user.
    # For now, we drop them automatically to prevent perfect models.
    df_modeling = df.copy()
    dropped_for_leakage = []
    for warning in leakage_warnings:
        feat = warning["feature"]
        if feat in df_modeling.columns:
            df_modeling = df_modeling.drop(columns=[feat])
            dropped_for_leakage.append(feat)
            
    candidate_models = select_candidate_models(df_modeling, task_type)
    
    if not candidate_models:
        return {"error": "No suitable candidate models found for this dataset."}
        
    benchmark_results = train_and_evaluate(df_modeling, target_col, task_type, candidate_models)
    
    return {
        "task_type": task_type,
        "leakage_warnings": leakage_warnings,
        "dropped_features": dropped_for_leakage,
        "candidates": candidate_models,
        "benchmark_results": benchmark_results
    }
