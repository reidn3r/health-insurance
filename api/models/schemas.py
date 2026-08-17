from typing import Literal, Union

from pydantic import BaseModel, Field


class InsuranceRecord(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: int = Field(ge=18, le=100)
    Driving_License: int = Field(ge=0, le=1)
    Region_Code: Union[int, str]
    Previously_Insured: int = Field(ge=0, le=1)
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(gt=0)
    Policy_Sales_Channel: Union[int, str]
    Vintage: int = Field(ge=0, le=300)


class Prediction(BaseModel):
    response_score: float
    probability_no: float
    prediction: int


class BatchPrediction(BaseModel):
    predictions: list[Prediction]