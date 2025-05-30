from typing import Dict

from psrc.core.interfaces.i_card_deck import ICardDeck


class CardDeck(ICardDeck):
    """
    CardDeck is an implementation of the ICardDeck interface.

    This implementation initializes a combined multi-deck count, and provides methods to add or remove a card by
    normalizing face cards to label 9 and updating the internal counts accordingly.
    """

    def __init__(self, deck_count: int) -> None:
        """
        Initialize CardDeck with the specified number of combined decks.

        This sets counts for labels 0–8 to 4 * deck_count each, and for label 9 (10/J/Q/K) to 16 * deck_count.

        Parameters:
            deck_count (int): The number of decks to combine.
        """
        self.cards: Dict[int, int] = {i: 4 * deck_count for i in range(0, 9)}
        self.cards[9] = 16 * deck_count

    def add_card(self, card_label: int) -> bool:
        """
        Add a card to the deck.

        This implementation first normalizes any face-card label (9–12) to 9, then increments the count if that
        label exists.

        Parameters:
            card_label (int): The numeric label of the card to be added.

        Returns:
            bool: True if the card was successfully added, False otherwise.
        """
        normalized_label = 9 if card_label in {9, 10, 11, 12} else card_label

        if normalized_label in self.cards:
            self.cards[normalized_label] += 1
            return True
        else:
            return False

    def remove_card(self, card_label: int) -> bool:
        """
        Remove a card from the deck.

        This implementation first normalizes any face-card label (9–12) to 9, then decrements the count if that
        label exists.

        Parameters:
            card_label (int): The numeric label of the card to be removed.

        Returns:
            bool: True if the card was successfully removed, False otherwise.
        """
        normalized_label = 9 if card_label in {9, 10, 11, 12} else card_label

        if normalized_label in self.cards and self.cards[normalized_label] > 0:
            self.cards[normalized_label] -= 1
            return True
        else:
            return False
