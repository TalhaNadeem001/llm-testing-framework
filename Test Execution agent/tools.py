def validateRequestedItem(item_name: str, details: str = "") -> dict:
    return {"valid": True, "item_name": item_name, "details": details, "message": f"'{item_name}' is available on the menu."}


def addItemsToOrder(item_name: str, quantity: int = 1, details: str = "", dine_in: bool = False) -> dict:
    return {
        "success": True,
        "item_name": item_name,
        "quantity": quantity,
        "details": details,
        "dine_in": dine_in,
        "message": f"Added {quantity}x {item_name} to the order.",
    }


def suggestedPickupTime(time: str) -> dict:
    return {"confirmed": True, "pickup_time": time, "message": f"Pickup time set to {time}."}


def saveHumanName(name: str) -> dict:
    return {"success": True, "name": name, "message": f"Customer name saved as '{name}'."}


def humanInterventionNeeded(reason: str) -> dict:
    return {"flagged": True, "reason": reason, "message": "Cashier has been notified to assist."}


def getMenuLink() -> dict:
    return {"url": "https://example-restaurant.com/menu", "message": "Here is the link to our full menu."}


def calcOrderPrice() -> dict:
    return {
        "subtotal": 18.50,
        "tax": 1.76,
        "total": 20.26,
        "currency": "USD",
        "message": "Order total calculated.",
    }


def askingForPickupTime() -> dict:
    return {"pickup_time": "3:45 PM", "confirmed": True, "message": "Confirmed pickup time is 3:45 PM."}


def askingForWaitTime() -> dict:
    return {"wait_time_minutes": 15, "message": "Current estimated wait time is 15 minutes."}


def getPreviousOrdersDetails(limit: int = 1) -> dict:
    orders = [
        {
            "order_id": "ORD-1041",
            "date": "2026-05-01",
            "items": [{"name": "Cheeseburger", "quantity": 1}, {"name": "Fries", "quantity": 1}],
            "total": 12.75,
        },
        {
            "order_id": "ORD-1038",
            "date": "2026-04-28",
            "items": [{"name": "Chicken Sandwich", "quantity": 2}],
            "total": 17.00,
        },
    ]
    return {"orders": orders[:limit], "count": min(limit, len(orders))}


def updateItemInOrder(item_name: str, details: str) -> dict:
    return {"success": True, "item_name": item_name, "new_details": details, "message": f"'{item_name}' has been updated with: {details}."}


def removeItemFromOrder(item_name: str) -> dict:
    return {"success": True, "item_name": item_name, "message": f"'{item_name}' has been removed from the order."}


def changeItemQuantity(item_name: str, quantity: int) -> dict:
    return {"success": True, "item_name": item_name, "new_quantity": quantity, "message": f"Quantity for '{item_name}' changed to {quantity}."}


def cancelOrder() -> dict:
    return {"cancelled": True, "message": "The order has been cancelled."}


def getOrderLineItems() -> dict:
    return {
        "items": [
            {"name": "Cheeseburger", "quantity": 1, "details": "no pickles", "price": 9.00},
            {"name": "Fries", "quantity": 2, "details": "large", "price": 4.75},
        ],
        "item_count": 2,
    }


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "validateRequestedItem",
            "description": "Validate that a menu item exists and is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "addItemsToOrder",
            "description": "Add an item to the customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "details": {"type": "string"},
                    "dine_in": {"type": "boolean"},
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggestedPickupTime",
            "description": "Set or update the pickup time for the order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                },
                "required": ["time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "saveHumanName",
            "description": "Save the customer's name for the order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "humanInterventionNeeded",
            "description": "Flag the conversation for cashier or human intervention.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getMenuLink",
            "description": "Retrieve the URL link to the restaurant's menu.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcOrderPrice",
            "description": "Calculate the total price of the current order.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "askingForPickupTime",
            "description": "Retrieve the confirmed pickup time for an existing order.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "askingForWaitTime",
            "description": "Get the current estimated wait time when no confirmed order exists.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getPreviousOrdersDetails",
            "description": "Retrieve details of the customer's previous orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "updateItemInOrder",
            "description": "Update details or customization of an item already in the order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": ["item_name", "details"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "removeItemFromOrder",
            "description": "Remove an item from the customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "changeItemQuantity",
            "description": "Change the quantity of an item in the order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["item_name", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancelOrder",
            "description": "Cancel the customer's current order.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getOrderLineItems",
            "description": "Retrieve all line items currently in the customer's order.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


if __name__ == "__main__":
    print(validateRequestedItem("Cheeseburger"))
    print(addItemsToOrder("Fries", quantity=2, details="large"))
    print(suggestedPickupTime("3:45 PM"))
    print(saveHumanName("Alice"))
    print(humanInterventionNeeded("Customer is upset about a missing item."))
    print(getMenuLink())
    print(calcOrderPrice())
    print(askingForPickupTime())
    print(askingForWaitTime())
    print(getPreviousOrdersDetails(limit=2))
    print(updateItemInOrder("Cheeseburger", "extra cheese"))
    print(removeItemFromOrder("Fries"))
    print(changeItemQuantity("Cheeseburger", 3))
    print(cancelOrder())
    print(getOrderLineItems())
