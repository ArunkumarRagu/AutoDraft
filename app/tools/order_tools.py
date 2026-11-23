
# Temporary sample data for learning

ORDERS = {
    1024: {
        "order_id": 1024,
        "customer_name": "Arun",
        "status": "Shipped",
        "total_amount": 2499,
    },
    1025: {
        "order_id": 1025,
        "customer_name": "John",
        "status": "Processing",
        "total_amount": 1599,
    },
}


def get_order_status(order_id: int) -> dict:

    """
    Retrieve the status of an order.

    Args:
        order_id: Unique order identifier.

    Returns:
        Order details if found, otherwise an error.
    """

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"Order {order_id} was not found."
        }

    return {
        "success": True,
        "data": order
    }
