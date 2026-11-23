
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.agents.email_schemas import (
    EmailUnderstanding,
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


SYSTEM_PROMPT = """
You are an AI customer support email understanding assistant.

Your task is to understand incoming customer emails
and extract their primary question.

Classify emails using these rules:

1. general_support:
   Questions about general company policies,
   product usage, subscriptions, FAQs and procedures.

2. data_question:
   Questions requiring customer-specific information
   from the application database, such as order history,
   order status, invoices, payments or account records.

3. needs_clarification:
   Use when the customer's intent cannot be determined
   from the email.

Additional rules:

- Extract the customer's name only if explicitly available.
- Never invent customer information.
- Extract the main question without changing its meaning.
- Select the most relevant category.
- Do not answer the customer's question.
- Do not assume access to company data.
"""


def understand_email(email_content: str) -> EmailUnderstanding:

    response = client.responses.parse(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=email_content,
        text_format=EmailUnderstanding,
    )

    result = response.output_parsed

    if result is None:
        raise ValueError(
            "The model did not return a valid structured result"
        )

    return result
