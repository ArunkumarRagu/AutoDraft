
from app.agents.email_understanding import understand_email
from app.agents.order_agent import run_order_agent


def process_customer_email(email_content: str):

    # STEP 1: Understand incoming email

    email = understand_email(email_content)

    print("\n--- EMAIL UNDERSTANDING ---")
    print("Customer:", email.customer_name)
    print("Intent:", email.intent.value)
    print("Question:", email.question)
    print("Category:", email.category.value)

    # STEP 2: Route based on intent

    if email.intent.value == "data_question":

        print("\nRouting to Order Agent...")

        response = run_order_agent(email.question)

        return response

    elif email.intent.value == "general_support":

        print("\nRouting to Knowledge RAG Agent...")

        # We will implement this in Milestone 2.
        return "Knowledge RAG Agent is not implemented yet."

    else:

        return (
            "We need additional information "
            "to understand your question."
        )


if __name__ == "__main__":

    sample_email = """
    Subject: Order status

    Hi Support Team,

    what is the use of the product I purchased?

    Thanks,
    Arun
    """

    result = process_customer_email(sample_email)

    print("\n--- FINAL RESPONSE ---")
    print(result)
