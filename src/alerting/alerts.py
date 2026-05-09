from src.utils.config import load_config


def generate_alert(score):
    config = load_config().get("alerting", {})
    high_risk_score = config.get("high_risk_score", 70)
    medium_risk_score = config.get("medium_risk_score", 40)

    if score > high_risk_score:
        return "HIGH OUTBREAK RISK"

    if score > medium_risk_score:
        return "MEDIUM OUTBREAK RISK"

    return "LOW OUTBREAK RISK"

