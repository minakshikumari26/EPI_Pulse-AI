import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def generate_alert(cases=None, risk_score=None, z_score=None, growth_rate=None):
    config = load_config().get("alerting", {})
    high_risk_score = config.get("high_risk_score", 70)
    medium_risk_score = config.get("medium_risk_score", 40)
    spike_z_score = config.get("spike_z_score", 1.5)
    growth_rate_warning = config.get("growth_rate_warning", 0.2)

    if risk_score is not None:
        if risk_score >= high_risk_score:
            return "High outbreak risk"
        if risk_score >= medium_risk_score:
            return "Medium outbreak risk"

    if z_score is not None and z_score >= spike_z_score:
        return "High outbreak risk"

    if growth_rate is not None and growth_rate >= growth_rate_warning:
        return "Medium outbreak risk"

    if cases is not None:
        if cases > 50:
            return "High outbreak risk"
        if cases > 30:
            return "Medium outbreak risk"

    return "Low outbreak risk"


def generate_alert_with_explanation(
    cases: Optional[float] = None,
    risk_score: Optional[float] = None,
    z_score: Optional[float] = None,
    growth_rate: Optional[float] = None,
    region: str = "Unknown",
    use_llm: bool = True,
) -> Dict[str, str]:
    """
    Generate alert with optional LLM-powered explanation.

    Args:
        cases: Number of cases
        risk_score: Risk score (0-100)
        z_score: Z-score for anomaly detection
        growth_rate: Growth rate of cases
        region: Region name
        use_llm: Whether to use LLM for explanation

    Returns:
        Dictionary with alert and explanation
    """
    alert = generate_alert(
        cases=cases,
        risk_score=risk_score,
        z_score=z_score,
        growth_rate=growth_rate,
    )

    result = {
        "alert": alert,
        "region": region,
    }

    # Try to add LLM explanation if enabled
    if use_llm:
        try:
            config = load_config().get("llm", {})
            if config.get("enabled", False):
                from src.llm import get_llm_client

                llm = get_llm_client()
                explanation = llm.explain_alert(
                    cases=cases,
                    risk_score=risk_score,
                    z_score=z_score,
                    growth_rate=growth_rate,
                    region=region,
                )
                result["explanation"] = explanation
        except Exception:
            # Fallback: do not break alert generation if LLM fails
            result["explanation"] = "LLM unavailable. Please check Ollama service."

    return result


def generate_region_alert(row):
    return {
        "date": row.get("date"),
        "region": row.get("region"),
        "alert": generate_alert(
            cases=row.get("cases"),
            risk_score=row.get("risk_score"),
            z_score=row.get("z_score"),
            growth_rate=row.get("growth_rate"),
        ),
    }


def generate_region_alert_with_llm(row, use_llm: bool = True) -> Dict:
    """
    Generate region alert with optional LLM explanation.

    Args:
        row: Row data with date, region, and metrics
        use_llm: Whether to use LLM for explanation

    Returns:
        Dictionary with alert and explanation
    """
    return generate_alert_with_explanation(
        cases=row.get("cases"),
        risk_score=row.get("risk_score"),
        z_score=row.get("z_score"),
        growth_rate=row.get("growth_rate"),
        region=row.get("region", "Unknown"),
        use_llm=use_llm,
    )


if __name__ == "__main__":
    print(generate_alert(cases=45, risk_score=62))
