from typing import List, Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input schema: one customer record with all raw features."""

    gender: str = Field(..., description="Female or Male")
    senior_citizen: int = Field(..., description="0 = no, 1 = yes", ge=0, le=1)
    partner: str = Field(..., description="Yes or No")
    dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Months with the company", ge=0, le=120)
    phone_service: str = Field(..., description="Yes or No")
    multiple_lines: str = Field(..., description="No, No phone service, or Yes")
    internet_service: str = Field(..., description="DSL, Fiber optic, or No")
    online_security: str = Field(..., description="No, No internet service, or Yes")
    online_backup: str = Field(..., description="No, No internet service, or Yes")
    device_protection: str = Field(..., description="No, No internet service, or Yes")
    tech_support: str = Field(..., description="No, No internet service, or Yes")
    streaming_tv: str = Field(..., description="No, No internet service, or Yes")
    streaming_movies: str = Field(..., description="No, No internet service, or Yes")
    contract: str = Field(..., description="Month-to-month, One year, or Two year")
    paperless_billing: str = Field(..., description="Yes or No")
    payment_method: str = Field(
        ...,
        description="Bank transfer (automatic), Credit card (automatic), Electronic check, or Mailed check",
    )
    monthly_charges: float = Field(..., description="Current monthly charge", ge=0)
    total_charges: Optional[float] = Field(
        None, description="Total charges to date (blank for brand-new customers)", ge=0
    )


class PredictionResponse(BaseModel):
    """Output schema for a single prediction."""

    churn_proba: float = Field(..., description="Predicted probability of churning")
    churn_flag: int = Field(..., description="0 = predicted to stay, 1 = predicted to churn")


class BatchPredictionResponse(BaseModel):
    """Output schema for a batch of predictions."""

    predictions: List[PredictionResponse]
    threshold: float = Field(..., description="Probability threshold used for the binary flag")
