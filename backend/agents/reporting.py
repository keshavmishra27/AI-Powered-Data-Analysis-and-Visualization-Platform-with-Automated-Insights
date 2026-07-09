from backend.core.evaluation import generate_leaderboard
from backend.llm.prompts import get_confidence_report_prompt
from backend.llm.providers import query_llm_json

def run_reporting_agent(benchmark_results: list, task_type: str, dataset_metrics: dict) -> dict:
    """
    Executes the Reporting Agent logic.
    1. Generates the leaderboard.
    2. Queries LLM for Confidence Report.
    """
    leaderboard = generate_leaderboard(benchmark_results)
    
    if not leaderboard:
        return {"error": "No benchmark results to report."}
        
    prompt = get_confidence_report_prompt(leaderboard, task_type, dataset_metrics)
    
    fallback_report = {
        "confidence_level": "Medium",
        "reasons": ["Automated benchmark completed successfully."],
        "warnings": ["LLM evaluation failed, using fallback report."],
        "recommendation": f"Proceed with {leaderboard[0]['model']} cautiously."
    }
    
    confidence_report = query_llm_json(prompt, fallback_report)
    
    return {
        "leaderboard": leaderboard,
        "confidence_report": confidence_report
    }
