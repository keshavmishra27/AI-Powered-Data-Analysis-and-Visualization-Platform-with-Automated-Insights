import pandas as pd
from backend.core.cleaning import generate_cleaning_suggestions, apply_cleaning
from backend.core.profiling import calculate_data_quality_score

def run_data_preparation_agent(df: pd.DataFrame, approved_actions: list = None) -> dict:
    """
    Executes Data Preparation Agent logic.
    If `approved_actions` is None, it generates suggestions.
    If `approved_actions` is provided, it applies them and returns the new DF and the DQ shift.
    """
    initial_dq = calculate_data_quality_score(df)
    
    if approved_actions is None:
        suggestions = generate_cleaning_suggestions(df)
        return {
            "state": "pending_approval",
            "initial_quality_score": initial_dq,
            "suggestions": suggestions
        }
    else:
        cleaned_df = apply_cleaning(df, approved_actions)
        final_dq = calculate_data_quality_score(cleaned_df)
        
        return {
            "state": "cleaned",
            "cleaned_df": cleaned_df,
            "initial_quality_score": initial_dq,
            "final_quality_score": final_dq,
            "quality_shift": round(final_dq["overall"] - initial_dq["overall"], 1)
        }
