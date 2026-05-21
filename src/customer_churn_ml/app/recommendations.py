"""Business recommendation engine for churn explainability.

Maps the top churn-driving features (from per-user SHAP values) to
actionable, user-friendly business recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Display-name mapping: internal engineered feature name → human-readable label
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "tenure": "Tenure (Months)",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "senior_citizen": "Senior Citizen Status",
    "num_services": "Number of Add-on Services",
    "avg_monthly_charge": "Average Monthly Charge",
    "multiple_lines_No phone service": "Multiple Lines — No Phone Service",
    "multiple_lines_Yes": "Multiple Lines",
    "internet_service_Fiber optic": "Internet Service — Fiber Optic",
    "internet_service_No": "No Internet Service",
    "online_security_No internet service": "Online Security — No Internet",
    "online_security_Yes": "Online Security",
    "online_backup_No internet service": "Online Backup — No Internet",
    "online_backup_Yes": "Online Backup",
    "device_protection_No internet service": "Device Protection — No Internet",
    "device_protection_Yes": "Device Protection",
    "tech_support_No internet service": "Tech Support — No Internet",
    "tech_support_Yes": "Tech Support",
    "streaming_tv_No internet service": "Streaming TV — No Internet",
    "streaming_tv_Yes": "Streaming TV",
    "streaming_movies_No internet service": "Streaming Movies — No Internet",
    "streaming_movies_Yes": "Streaming Movies",
    "contract_One year": "One-Year Contract",
    "contract_Two year": "Two-Year Contract",
    "payment_method_Credit card (automatic)": "Payment — Credit Card (Auto)",
    "payment_method_Electronic check": "Payment — Electronic Check",
    "payment_method_Mailed check": "Payment — Mailed Check",
    "tenure_group_13-24": "Tenure Group 13–24 Months",
    "tenure_group_25-48": "Tenure Group 25–48 Months",
    "tenure_group_49-72": "Tenure Group 49–72 Months",
    "gender": "Gender",
    "partner": "Partner Status",
    "dependents": "Dependents Status",
    "phone_service": "Phone Service",
    "paperless_billing": "Paperless Billing",
    "has_internet": "Has Internet",
    "has_phone": "Has Phone",
}


def _get_display_name(feature_name: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


# ---------------------------------------------------------------------------
# Recommendation rules
# Each rule is a callable: (feature_name, raw_row: pd.Series) → Optional[str]
# Return None when the rule does not apply.
# ---------------------------------------------------------------------------


def _recommend_tenure(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "tenure":
        return None
    v = raw_row.get("tenure")
    if v is None or pd.isna(v):
        return None
    v = float(v)
    if v < 6:
        return "Invest in early-stage onboarding and retention touchpoints for new customers in their first 3–6 months."
    if v < 12:
        return "Strengthen engagement programs for customers in their first year to build loyalty."
    return "Leverage tenure-based loyalty rewards to reinforce long-term commitment."


def _recommend_contract(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    contract = raw_row.get("contract")
    if contract is None:
        return None
    if feature_name == "contract_One year":
        if contract != "One year":
            return "Offer a one-year contract upgrade to reduce churn risk and increase commitment."
        return None
    if feature_name == "contract_Two year":
        if contract != "Two year":
            return "Offer a two-year contract with a loyalty discount to lock in long-term retention."
        return None
    return None


def _recommend_internet_service(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    internet = raw_row.get("internet_service")
    if internet is None:
        return None
    if feature_name == "internet_service_Fiber optic":
        if internet == "Fiber optic":
            return "Investigate fibre optic service quality or pricing concerns; consider offering a DSL downgrade or service credits."
        return None
    if feature_name == "internet_service_No":
        if internet == "No":
            return "Cross-sell internet service bundles to increase engagement and switching costs."
        return None
    return None


def _recommend_payment_method(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    payment = raw_row.get("payment_method")
    if payment is None:
        return None
    if feature_name == "payment_method_Electronic check":
        if payment == "Electronic check":
            return "Migrate the customer to automatic payment (credit card or bank transfer) to improve payment reliability and retention."
        return None
    if feature_name == "payment_method_Mailed check":
        if payment == "Mailed check":
            return "Encourage switching from mailed check to automatic digital payment for convenience and lower churn risk."
        return None
    return None


def _recommend_monthly_charges(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "monthly_charges":
        return None
    v = raw_row.get("monthly_charges")
    if v is None or pd.isna(v):
        return None
    if float(v) > 80:
        return "Review the pricing plan; consider a loyalty discount or plan optimization to improve perceived value."
    return None


def _recommend_total_charges(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "total_charges":
        return None
    v = raw_row.get("total_charges")
    if v is None or pd.isna(v):
        return None
    if float(v) < 100:
        return "New customer — prioritize proactive retention touchpoints in the first 3–6 months."
    return None


def _recommend_online_security(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name == "online_security_Yes":
        if raw_row.get("online_security") != "Yes":
            return "Offer online security as an add-on to increase perceived value and reduce churn risk."
        return None
    return None


def _recommend_tech_support(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name == "tech_support_Yes":
        if raw_row.get("tech_support") != "Yes":
            return "Offer tech support add-on; customers with support coverage are significantly less likely to churn."
        return None
    return None


def _recommend_num_services(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "num_services":
        return None
    v = raw_row.get("num_services")
    if v is None or pd.isna(v):
        return None
    if float(v) < 2:
        return "Cross-sell additional services (streaming, backup, security) to deepen engagement and raise switching costs."
    return None


def _recommend_paperless_billing(
    feature_name: str, raw_row: pd.Series
) -> Optional[str]:
    if feature_name != "paperless_billing":
        return None
    if raw_row.get("paperless_billing") == "Yes":
        return "Ensure the paperless billing experience is smooth; offer billing support or alternative notification channels."
    return None


def _recommend_senior_citizen(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "senior_citizen":
        return None
    if raw_row.get("senior_citizen") == 1:
        return "Senior customer — offer a dedicated support channel and simplified plan options tailored to their needs."
    return None


def _recommend_multiple_lines(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name == "multiple_lines_Yes":
        if raw_row.get("multiple_lines") != "Yes":
            return "Offer multiple lines to increase household switching costs and improve retention."
        return None
    return None


def _recommend_streaming_tv(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name == "streaming_tv_Yes":
        if raw_row.get("streaming_tv") != "Yes":
            return "Cross-sell streaming TV to increase service stickiness and monthly engagement."
        return None
    return None


def _recommend_streaming_movies(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name == "streaming_movies_Yes":
        if raw_row.get("streaming_movies") != "Yes":
            return "Cross-sell streaming movies to deepen content engagement and reduce churn likelihood."
        return None
    return None


def _recommend_online_backup(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name == "online_backup_Yes":
        if raw_row.get("online_backup") != "Yes":
            return "Offer online backup add-on to increase perceived value and service stickiness."
        return None
    return None


def _recommend_device_protection(
    feature_name: str, raw_row: pd.Series
) -> Optional[str]:
    if feature_name == "device_protection_Yes":
        if raw_row.get("device_protection") != "Yes":
            return "Offer device protection add-on as a low-friction upsell to improve retention."
        return None
    return None


def _recommend_has_internet(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "has_internet":
        return None
    if raw_row.get("has_internet") == 0:
        return "Customer has no internet — bundle internet with phone to increase overall engagement."
    return None


def _recommend_has_phone(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    if feature_name != "has_phone":
        return None
    if raw_row.get("has_phone") == 0:
        return "Customer has no phone service — offer phone service bundles to raise switching costs."
    return None


def _recommend_tenure_group(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    group = raw_row.get("tenure_group")
    if group is None:
        return None
    if feature_name == "tenure_group_13-24" and group == "13-24":
        return "Customer is in the 13–24 month window — introduce loyalty perks before the second-year decision point."
    if feature_name == "tenure_group_25-48" and group == "25-48":
        return "Mid-tenure customer — review plan fit and proactively offer contract renewal incentives."
    return None


# Ordered list of recommendation rule functions
_RECOMMENDATION_RULES = [
    _recommend_contract,
    _recommend_tenure,
    _recommend_internet_service,
    _recommend_payment_method,
    _recommend_monthly_charges,
    _recommend_total_charges,
    _recommend_online_security,
    _recommend_tech_support,
    _recommend_num_services,
    _recommend_paperless_billing,
    _recommend_senior_citizen,
    _recommend_multiple_lines,
    _recommend_streaming_tv,
    _recommend_streaming_movies,
    _recommend_online_backup,
    _recommend_device_protection,
    _recommend_has_internet,
    _recommend_has_phone,
    _recommend_tenure_group,
]


def generate_recommendation(feature_name: str, raw_row: pd.Series) -> Optional[str]:
    """Return a single actionable recommendation for a feature/raw-row pair, or None.

    Rules are evaluated in order; the first non-None result wins.
    """
    for rule in _RECOMMENDATION_RULES:
        rec = rule(feature_name, raw_row)
        if rec is not None:
            return rec
    return None


def build_recommendations(
    top_features: List[Tuple[str, float, Any]],
    raw_row: pd.Series,
) -> List[Dict[str, Any]]:
    """Convert top churn-driving features into a list of recommendation dicts.

    Args:
        top_features: List of (feature_name, shap_value, feature_value) tuples,
            sorted by descending shap_value (most churn-driving first).
        raw_row: The original (pre-preprocessing) feature row as a pandas Series,
            used by recommendation rules to look up raw categorical/numeric values.

    Returns:
        List of dicts with keys: feature, display_name, shap_value,
        feature_value, recommendation.
    """
    results: List[Dict[str, Any]] = []
    for feature_name, shap_value, feature_value in top_features:
        rec_text = generate_recommendation(feature_name, raw_row)
        if rec_text is None:
            # Fallback: if no specific rule matched, provide a generic insight
            rec_text = f"Review '{_get_display_name(feature_name)}' as it is contributing to churn risk."
        results.append(
            {
                "feature": feature_name,
                "display_name": _get_display_name(feature_name),
                "shap_value": round(float(shap_value), 6),
                "feature_value": feature_value,
                "recommendation": rec_text,
            }
        )
    return results
