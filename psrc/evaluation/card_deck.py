from typing import Dict

from psrc.core.interfaces.i_card_deck import ICardDeck


class CardDeck(ICardDeck):
    """
    CardDeck implements the ICardDeck interface for managing a blackjack card deck.

    This class provides methods for adding and removing cards from the deck, updating the deck's state to reflect
    gameplay changes. The deck is initialized with a specified deck_count, where the count of each card is scaled
    based on the number of decks.
    """

    def __init__(self, deck_count: int) -> None:
        """
        Initialize the CardDeck with a given deck count.

        Parameters:
          deck_count (int): The number of decks to be combined.

        The deck is initialized with:
          - 4 * deck_count copies of cards labeled 0 through 8.
          - 16 * deck_count copies for card label 9 (face cards are normalized to label 9).
        """
        # Initialize counts for cards 0 through 8 (each has 4 * deck_count copies)
        self.cards: Dict[int, int] = {i: 4 * deck_count for i in range(0, 9)}
        # Initialize counts for face cards (normalized to label 9, each has 16 * deck_count copies)
        self.cards[9] = 16 * deck_count

    def add_card(self, card_label: int) -> bool:
        """
        Add a card to the deck.

        The card_label is normalized so that any face card (labels 9, 10, 11, or 12) is treated as 9.

        Parameters:
          card_label (int): The numeric label of the card to be added.

        Returns:
          bool: True if the card was successfully added, False otherwise.
        """
        # Normalize the card label: face cards (9, 10, 11, 12) are considered as label 9
        normalized_label = 9 if card_label in {9, 10, 11, 12} else card_label

        if normalized_label in self.cards:
            self.cards[normalized_label] += 1
            return True
        else:
            return False

    def remove_card(self, card_label: int) -> bool:
        """
        Remove a card from the deck.

        The card_label is normalized so that any face card (labels 9, 10, 11, or 12) is treated as 9.

        Parameters:
          card_label (int): The numeric label of the card to be removed.

        Returns:
          bool: True if the card was successfully removed, False otherwise.
        """
        # Normalize the card label: face cards (9, 10, 11 ,12) are considered as label 9
        normalized_label = 9 if card_label in {9, 10, 11, 12} else card_label

        if normalized_label in self.cards and self.cards[normalized_label] > 0:
            self.cards[normalized_label] -= 1
            return True
        else:
            return False
