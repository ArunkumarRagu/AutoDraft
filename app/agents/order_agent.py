
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from app.tools.order_tools import get_order_status


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# Tool definitions exposed to the LLM

ORDER_TOOLS = [
    {
        "type": "function",
        "name": "get_order_status",
        "description": (
            "Retrieve the status and details of a customer order. "
            "Use this when the customer asks about a specific "
            "order and provides its order ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "Unique order ID"
                }
            },
            "required": ["order_id"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# Map model-visible tool names to Python functions

AVAILABLE_FUNCTIONS = {
    "get_order_status": get_order_status
}


def run_order_agent(question: str):

    instructions = """
    You are a customer support assistant.

    You can retrieve order information using the
    get_order_status tool.

    When a customer asks about a specific order,
    use the tool to retrieve its actual status.

    Never invent order information.

    If an order ID is missing, ask the customer
    to provide it.

    Base your answer on the tool result.
    """

    # STEP 1: Send question and available tools

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=question,
        tools=ORDER_TOOLS,
        tool_choice="auto"
    )

    # STEP 2: Find tool calls requested by the model

    tool_calls = [
        item for item in response.output
        if item.type == "function_call"
    ]

    if not tool_calls:
        return response.output_text

    # STEP 3: Execute requested Python functions

    tool_outputs = []

    for tool_call in tool_calls:

        function_name = tool_call.name

        arguments = json.loads(tool_call.arguments)

        function = AVAILABLE_FUNCTIONS.get(function_name)

        if function is None:
            raise ValueError(
                f"Unknown tool: {function_name}"
            )

        result = function(**arguments)

        print(f"\nExecuting tool: {function_name}")
        print("Arguments:", arguments)
        print("Result:", result)

        # STEP 4: Prepare tool result for the LLM

        tool_outputs.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result)
        })

    # STEP 5: Send results back to the LLM

    final_response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        previous_response_id=response.id,
        input=tool_outputs,
        tools=ORDER_TOOLS,
        tool_choice="auto"
    )

    return final_response.output_text
