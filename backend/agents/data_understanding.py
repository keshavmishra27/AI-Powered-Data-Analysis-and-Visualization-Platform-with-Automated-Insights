import pandas as pd
from backend.core.profiling import detect_schema, calculate_data_quality_score, check_dataset_suitability
from backend.llm.prompts import get_schema_explanation_prompt
from backend.llm.providers import query_llm_json

def run_understanding_agent(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Executes the Data Understanding Agent logic.
    1. Detects schema.
    2. Calculates Data Quality Score.
    3. Checks dataset suitability.
    4. Queries LLM for human-readable explanation of schema implications.
    """
    # 1. Deterministic Schema Detection
    schema = detect_schema(df)
    
    # 2. Data Quality Score
    quality_score = calculate_data_quality_score(df)
    
    # 3. Suitability check
    suitability = check_dataset_suitability(df, target_col)
    
    # 4. LLM Explanation
    prompt = get_schema_explanation_prompt(schema, quality_score)
    fallback_explanation = {
        "summary": f"The dataset has {len(df)} rows and {len(df.columns)} columns with an overall quality score of {quality_score['overall']}/100.",
        "implications": ["Standard data cleaning recommended before modeling."]
    }
    
    explanation = query_llm_json(prompt, fallback=fallback_explanation)
    
    return {
        "schema": schema,
        "quality_score": quality_score,
        "suitability": suitability,
        "explanation": explanation
    }
