def format_currency(amount, symbol="$"):
    """Format a numeric amount as a currency string with a leading symbol."""
    return f"{symbol}{amount:.2f}"
