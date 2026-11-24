
from app.agents.email_understanding import understand_email
from app.agents.order_agent import run_order_agent
from app.rag.generation import generate_support_response


def process_customer_email(email_content: str):
    # Step 1: Understand the incoming email
    email = understand_email(email_content)

    print("\n--- EMAIL UNDERSTANDING ---")
    print("Customer:", email.customer_name)
    print("Intent:", email.intent.value)
    print("Question:", email.question)
    print("Category:", email.category.value)

    # Step 2: Route based on the detected intent
    if email.intent.value == "data_question":
        print("\nRouting to Order Agent...")
        return run_order_agent(email.question)

    elif email.intent.value == "general_support":
        print("\nRouting to Knowledge RAG Agent...")
        return generate_support_response(email.question)

    else:
        return (
            "Could you please provide some additional "
            "details so that our support team can help you?"
        )


if __name__ == "__main__":
    sample_email = """
    Subject: Refund request

    Hi Support Team,

    can i able to join as a seller with your organisation?

    Thanks,
    Arun
    """

    result = process_customer_email(sample_email)

    print("\n--- FINAL RESPONSE ---")
    print(result)
