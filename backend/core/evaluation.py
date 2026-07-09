def generate_leaderboard(benchmark_results: list) -> list:
    """
    Takes the raw benchmark results and formats them into a leaderboard.
    """
    if not benchmark_results:
        return []
        
    leaderboard = []
    for rank, res in enumerate(benchmark_results, start=1):
        leaderboard.append({
            "rank": rank,
            "model": res["model"],
            "score": round(res["score"], 4),
            "train_time_sec": res["train_time"],
            "inference_time_sec": res["inference_time"],
            "top_features": list(res["feature_importance"].keys())[:3]
        })
        
    return leaderboard
