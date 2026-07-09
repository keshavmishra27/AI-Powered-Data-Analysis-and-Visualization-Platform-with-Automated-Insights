# Schemas for LLM JSON output to enforce grounded responses

CONFIDENCE_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence_level": {
            "type": "string",
            "enum": ["High", "Medium", "Low"],
            "description": "The overall confidence in the model's reliability."
        },
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of reasons supporting the confidence level (e.g., 'balanced classes', 'sufficient samples')."
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of potential issues or warnings (e.g., 'slightly imbalanced features')."
        },
        "recommendation": {
            "type": "string",
            "description": "A final recommendation sentence (e.g., 'Suitable for production prototype')."
        }
    },
    "required": ["confidence_level", "reasons", "warnings", "recommendation"]
}

SCHEMA_EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A 1-2 sentence human-readable summary of the detected schema."
        },
        "implications": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What this schema implies for downstream ML tasks (e.g., 'High number of categorical columns will require encoding')."
        }
    },
    "required": ["summary", "implications"]
}
