from typing import Any, Dict, List
from jpype import JInt, JArray, JClass


def deck_to_java_array(deck: Dict[int, int]) -> Any:
    """
    Convert a Python deck dictionary to a Java array of integers.

    The deck is expected to be a dictionary mapping card labels (0-9) to their counts. The function constructs an
    array of counts in the order of card labels from 0 to 9 and converts it into a Java integer array.

    Parameters:
      deck (Dict[int, int]): A dictionary mapping card labels to their counts.

    Returns:
      Any: A Java array of integers representing the deck counts.
    """
    order = list(range(0, 10))
    deck_values = [deck.get(i, 0) for i in order]
    # Convert the Python list to a Java integer array using JArray and JInt
    return JArray(JInt)([JInt(val) for val in deck_values])


def hand_to_java_array_list(hand: List[int]) -> Any:
    """
    Convert a Python list representing a hand of cards to a Java ArrayList of integers.

    The hand is expected to be a list of card labels. The function normalizes the card values according to
    blackjack rules. The normalized values are then added to a Java ArrayList.

    Parameters:
      hand (List[int]): A list of card labels (integer values).

    Returns:
      Any: A Java ArrayList containing the normalized card values.
    """
    # Get the Java ArrayList class and create an instance
    ArrayList = JClass("java.util.ArrayList")
    java_list = ArrayList()

    # Iterate through each card in the hand and normalize its value
    for card in hand:
        if card == 0:
            value = 1
        elif 1 <= card <= 8:
            value = card + 1
        else:
            value = 10

        # Add the normalized value to the Java ArrayList as a JInt
        java_list.add(JInt(value))

    return java_list
