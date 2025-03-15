from typing import Any, Dict, List

from psrc.detection.detection_utils import group_cards

class HandTracker:
  def __init__(self) -> None:
    self.hands_state: Dict[str, Any] = {}

  def _evaluate_hand(self, cards: List[int]) -> int:
    total = 0
    aces = 0

    for card in cards:
      if card == 0:
        total += 1
        aces += 1
      elif 1 <= card <= 8:
        total += card + 1
      else:
        total += 10

    while aces > 0 and total + 10 <= 21:
      total += 10
      aces -= 1

    return int(total)

  def update(self, detections: Dict[tuple, Dict[str, Any]]) -> Dict[str, Any]:
    locked_detections: Dict[tuple, Dict[str, Any]] = {
      bbox: data for bbox, data in detections.items() if data.get("locked", False)
    }

    boxes: List[tuple] = list(locked_detections.keys())
    groups = group_cards([list(b) for b in boxes])
    hands_info: Dict[str, Any] = {}

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

    player_groups = [group for group in groups if len(group) > 1]
    player_groups.sort(key=lambda group: min(boxes[idx][0] for idx in group) if group else 0)

    for i, group in enumerate(player_groups, start=1):
      cards = [
          locked_detections[boxes[idx]]["label"] 
          for idx in group if boxes[idx] in locked_detections
      ]
      score = self._evaluate_hand(cards)
      hand_boxes = [boxes[idx] for idx in group]
      hands_info[f"Player {i}"] = {"cards": cards, "score": score, "boxes": hand_boxes}

    self.hands_state = hands_info
    return hands_info