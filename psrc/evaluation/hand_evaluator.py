from typing import Any, Dict

from psrc.core.interfaces.i_hand_evaluator import IHandEvaluator
from psrc.core.interfaces.i_card_deck import ICardDeck
from psrc.core.interfaces.i_ev_calculator import IExpectedValueCalculator


class HandEvaluator(IHandEvaluator):
    """
    HandEvaluator implements the IHandEvaluator interface for evaluating grouped blackjack hands and determining
    optimal actions.

    Uses an ICardDeck to access the current deck state and an EVCalculator to compute expected values for stand,
    hit, double, split, and surrender.
    """

    def __init__(
        self, deck: ICardDeck, ev_calculator: IExpectedValueCalculator
    ) -> None:
        """
        Initialize the HandEvaluator with a deck and EV calculator.

        Parameters:
          deck (ICardDeck): The deck manager providing current card counts.
          ev_calculator (IEVCalculator): The calculator for computing EVs.
        """
        self.deck = deck
        self.ev_calc = ev_calculator

    def evaluate_hands(self, hands_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate each player's hand and determine the optimal action.

        This method iterates through provided hands, computes EVs for stand, hit, double, split, and surrender
        using the EVCalculator, identifies the action with the highest EV for each player hand, and returns a
        mapping of hand IDs to their results.

        Parameters:
          hands_info (Dict[str, Any]): A dictionary mapping hand identifiers to their details (e.g., cards, score,
          boxes).

        Returns:
          Dict[str, Any]: A dictionary of evaluation results, mapping hand identifiers to a structure containing
          EVs for each action and the best action.
        """
        results: Dict[str, Any] = {}
        dealer_cards = hands_info.get("Dealer", {}).get("cards", [])

        if not dealer_cards:
            return {}

        # Compute EVs for each player hand, skipping over the dealer
        for hand_id, info in hands_info.items():
            if hand_id == "Dealer":
                continue

            player_cards = info.get("cards", [])
            evs: Dict[str, float] = {}

            evs["stand"] = self.ev_calc.calculate_stand_ev(
                self.deck.cards, player_cards, dealer_cards
            )
            evs["hit"] = self.ev_calc.calculate_hit_ev(
                self.deck.cards, player_cards, dealer_cards
            )
            evs["double"] = self.ev_calc.calculate_double_ev(
                self.deck.cards, player_cards, dealer_cards
            )
            evs["split"] = self.ev_calc.calculate_split_ev(
                self.deck.cards, player_cards, dealer_cards
            )
            evs["surrender"] = self.ev_calc.calculate_surrender_ev(
                self.deck.cards, player_cards, dealer_cards
            )

            # Determine best available action based on highest EV
            best_action = max(evs, key=evs.get)
            results[hand_id] = {"evs": evs, "best_action": best_action}

        return results
