import json

def get_schema_explanation_prompt(schema_data: dict, data_quality_score: dict) -> str:
    return f"""
You are an expert Data Scientist. Analyze the following dataset schema and data quality score.
Do NOT invent details. Do NOT mention specific rows.

Schema Detected:
{json.dumps(schema_data, indent=2)}

Data Quality Score:
{json.dumps(data_quality_score, indent=2)}

Respond ONLY with valid JSON matching this schema:
{{
  "summary": "1-2 sentences summarizing the dataset structure and quality.",
  "implications": ["bullet 1", "bullet 2"]
}}
"""

def get_confidence_report_prompt(leaderboard: list, task_type: str, dataset_metrics: dict) -> str:
    return f"""
You are an expert ML Engineer. Analyze the model leaderboard and dataset metrics to provide a confidence report on the best model.

Task Type: {task_type}
Dataset Metrics:
{json.dumps(dataset_metrics, indent=2)}

Top Models:
{json.dumps(leaderboard[:3], indent=2)}

Respond ONLY with valid JSON matching this schema:
{{
  "confidence_level": "High" | "Medium" | "Low",
  "reasons": ["reason 1", "reason 2"],
  "warnings": ["warning 1"],
  "recommendation": "Final recommendation"
}}
"""
