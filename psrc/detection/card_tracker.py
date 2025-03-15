from typing import Any, Callable, Dict, Optional

from psrc.core.interfaces.i_card_tracker import ICardTracker
from psrc.detection.detection_utils import compute_overlap

class CardTracker(ICardTracker):
  def __init__(
    self,
    confidence_threshold: float,
    overlap_threshold: float,
    confirmation_frames: int,
    disappear_frames: int,
    on_lock_callback: Optional[Callable[[str], None]] = None
  ) -> None:
    self.confidence_threshold = confidence_threshold
    self.overlap_threshold = overlap_threshold
    self.confirmation_frames = confirmation_frames
    self.disappear_frames = disappear_frames
    self.on_lock_callback = on_lock_callback
    self.tracked_cards: Dict[tuple, Dict[str, Any]] = {}

  def update(self, detections: Dict[tuple, Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    old_tracked = self.tracked_cards.copy()
    new_tracked: Dict[tuple, Dict[str, Any]] = {}

    matched_prev_keys = set()

    for box, detection_info in detections.items():
      label = detection_info.get("label")
      confidence = detection_info.get("confidence")
      matched_prev = None

      for prev_box, info in old_tracked.items():
        if compute_overlap(list(box), list(prev_box)) >= self.overlap_threshold:
          matched_prev = prev_box
          break

      if matched_prev:
        info = old_tracked[matched_prev]

        if confidence >= self.confidence_threshold:
          info["stable_frames"] = info.get("stable_frames", 0) + 1
          info["missing_frames"] = 0
        else:
          info["missing_frames"] = info.get("missing_frames", 0) + 1

        if info["stable_frames"] >= self.confirmation_frames and not info.get("locked", False):
          if self.on_lock_callback:
            self.on_lock_callback(info["label"])

          info["locked"] = True

        new_box_key = box
        new_tracked[new_box_key] = info
        matched_prev_keys.add(matched_prev)
      else:
        new_tracked[box] = {
          "label": label,
          "confidence": confidence,
          "stable_frames": 1,
          "missing_frames": 0,
          "locked": False
        }

    for prev_box, info in old_tracked.items():
      if prev_box not in matched_prev_keys:
        info["missing_frames"] = info.get("missing_frames", 0) + 1

        if info["missing_frames"] < self.disappear_frames:
          new_tracked[prev_box] = info

    self.tracked_cards = new_tracked
    return new_tracked
