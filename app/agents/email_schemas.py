
from enum import Enum
from pydantic import BaseModel, Field


class EmailIntent(str, Enum):
    GENERAL_SUPPORT = "general_support"
    DATA_QUESTION = "data_question"
    NEEDS_CLARIFICATION = "needs_clarification"


class EmailCategory(str, Enum):
    SUBSCRIPTION = "subscription"
    ORDER = "order"
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    OTHER = "other"


class EmailUnderstanding(BaseModel):
    customer_name: str | None = Field(
        description="Customer name if explicitly available, otherwise null"
    )

    intent: EmailIntent = Field(
        description="Whether this is general support, a data question, or unclear"
    )

    question: str = Field(
        description="The main question extracted from the email"
    )

    category: EmailCategory = Field(
        description="The primary category of the customer question"
    )
