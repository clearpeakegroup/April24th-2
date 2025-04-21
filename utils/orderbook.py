from typing import List, Dict, Tuple

def detect_gaps(orderbook: List[Tuple[float, int, int]]) -> List[float]:
    """
    Detect price levels with 0 qty on both bid and ask.
    Args:
        orderbook: List of (price, bid_qty, ask_qty)
    Returns:
        List of price levels that are empty (0 qty on both sides)
    """
    gaps = []
    for price, bid_qty, ask_qty in orderbook:
        if bid_qty == 0 and ask_qty == 0:
            gaps.append(price)
    return gaps 