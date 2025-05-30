from typing import Any, Dict

from psrc.core.interfaces.i_card_deck import ICardDeck
from psrc.core.interfaces.i_ev_calculator import IExpectedValueCalculator
from psrc.core.interfaces.i_hand_evaluator import IHandEvaluator


class HandEvaluator(IHandEvaluator):
    """
    HandEvaluator is an implementation of the IHandEvaluator interface.

    This implementation queries the current deck state and uses an EV calculator to compute stand, hit, double,
    split, and surrender values for each player hand, then selects the highest expected value action.
    """

    def __init__(
        self, deck: ICardDeck, ev_calculator: IExpectedValueCalculator
    ) -> None:
        """
        Initialize HandEvaluator with a deck and an EV calculator.

        Parameters:
            deck (ICardDeck): The deck for current card counts.
            ev_calculator (IExpectedValueCalculator): The calculator for EV computations.
        """
        self.deck = deck
        self.ev_calc = ev_calculator

    def evaluate_hands(
        self, hands: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate each player's hand and select the optimal action.

        This implementation skips evaluation if no dealer hand is present, for each non-dealer hand computes
        EVs for stand, hit, double, split, and surrender by calling the EV calculator, and records the best
        action.

        Parameters:
            hands (Dict[str, Dict[str, Any]]): A mapping of hand IDs to their hand information.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of hand IDs to their evaluation information.
        """
        results: Dict[str, Any] = {}
        dealer_cards = hands.get("Dealer", {}).get("cards", [])

        if not dealer_cards:
            return {}

        # Compute EVs for each player hand, skipping over the dealer
        for hand_id, info in hands.items():
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
