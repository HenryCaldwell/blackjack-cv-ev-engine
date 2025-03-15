from typing import Any, Dict, List

from psrc.core.interfaces.i_hand_tracker import IHandTracker
from psrc.detection.detection_utils import group_cards

class HandTracker(IHandTracker):
  """
  HandTracker implements the IHandTracker interface for tracking and grouping cards into blackjack hands.

  This class groups card detections into hands and calculates the corresponding hand scores using a predefined
  evaluation algorithm. Dealer hands are assumed to consist of single detections, while player hands consist of 
  grouped detections.
  """
  
  def __init__(self) -> None:
    """
    Initialize the HandTracker with an empty state for hands.
    """
    # Dictionary to store currently tracked hands; key is player/dealer hand, value is hand information
    self.hands_state: Dict[str, Any] = {}

  def _evaluate_hand(self, cards: List[int]) -> int:
    """
    Evaluate the total score of a hand based on card values.

    Cards are evaluated using the following rules:
      - Ace (represented by 0) is initially counted as 1, with an option to add 10 if it doesn't bust.
      - Cards 1 through 8 are valued at card value + 1.
      - Cards 9 and above are counted as 10.

    Parameters:
      cards (List[int]): A list of card labels (integer values).

    Returns:
      int: The calculated score of the hand.
    """
    total = 0
    aces = 0

    for card in cards:
      if card == 0:
        total += 1
        # Count Ace as 1 initially
        aces += 1
      # Cards 1-8 have values card + 1
      elif 1 <= card <= 8:
        total += card + 1
      # Cards 9 and above count as 10
      else:
        total += 10

    # Adjust Ace value if 10 does not bust the hand
    while aces > 0 and total + 10 <= 21:
      total += 10
      aces -= 1

    return int(total)

  def update(self, detections: Dict[tuple, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update the hand tracker with new card detections and group them into hands.

    This method processes the locked detections, groups them using an overlap-based grouping algorithm, and
    evaluates the scores for dealer and player hands. Dealer hands are assumed to be those with a single
    detection, while player hands consist of multiple grouped detections.

    Parameters:
      detections (Dict[tuple, Dict[str, Any]]): A dictionary mapping bounding box coordinates to detection
      details.

    Returns:
      Dict[str, Any]: A dictionary mapping hand identifiers (as strings) to their hand details (e.g., cards,
      score, boxes).
    """
    # Filter detections to include only those that are locked
    locked_detections: Dict[tuple, Dict[str, Any]] = {
      bbox: data for bbox, data in detections.items() if data.get("locked", False)
    }

    boxes: List[tuple] = list(locked_detections.keys())
    groups = group_cards([list(b) for b in boxes])
    hands_info: Dict[str, Any] = {}

    # Identify groups with a single detection (assumed to be the dealer's hand)
    single_groups = [group for group in groups if len(group) == 1]
    dealer_indices = [idx for group in single_groups for idx in group]
    
    if dealer_indices:
      dealer_cards = [
        locked_detections[boxes[idx]]["label"] 
        for idx in dealer_indices if boxes[idx] in locked_detections
      ]
      dealer_score = self._evaluate_hand(dealer_cards)
      dealer_boxes = [boxes[idx] for idx in dealer_indices]
      hands_info["Dealer"] = {"cards": dealer_cards, "score": dealer_score, "boxes": dealer_boxes}

    # Identify groups with more than one detection (assumed to be player hands)
    player_groups = [group for group in groups if len(group) > 1]
    # Sort player groups by the x-coordinate of the leftmost box for consistent ordering
    player_groups.sort(key=lambda group: min(boxes[idx][0] for idx in group) if group else 0)

    for i, group in enumerate(player_groups, start=1):
      cards = [
          locked_detections[boxes[idx]]["label"] 
          for idx in group if boxes[idx] in locked_detections
      ]
      score = self._evaluate_hand(cards)
      hand_boxes = [boxes[idx] for idx in group]
      hands_info[f"Player {i}"] = {"cards": cards, "score": score, "boxes": hand_boxes}

    # Update the hands state with the new tracking state
    self.hands_state = hands_info
    return hands_info