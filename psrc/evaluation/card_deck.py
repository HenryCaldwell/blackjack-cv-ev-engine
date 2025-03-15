from typing import Dict

from psrc.core.interfaces.i_card_deck import ICardDeck

class CardDeck(ICardDeck):
  def __init__(self, size: int) -> None:
    self.size = size
    self.cards: Dict[int, int] = {i: 4 * size for i in range(0, 9)}
    self.cards[9] = 16 * size

  def add_card(self, card_label: int) -> bool:
    normalized_label = 9 if card_label in {9, 10, 11, 12} else card_label

    if normalized_label in self.cards:
      self.cards[normalized_label] += 1
      return True
    else:
      return False

  def remove_card(self, card_label: int) -> bool:
    normalized_label = 9 if card_label in {9, 10, 11, 12} else card_label

    if normalized_label in self.cards and self.cards[normalized_label] > 0:
      self.cards[normalized_label] -= 1
      return True
    else:
      return False