from typing import Any, Callable, Dict, Optional

from psrc.core.interfaces.i_card_tracker import ICardTracker
from psrc.detection.detection_utils import compute_overlap

class CardTracker(ICardTracker):
  """
  CardTracker implements the ICardTracker interface for tracking card detections across video frames.
  
  It updates tracking information by matching new detections with existing tracks using overlap criteria.
  stability of a detection is monitored over consecutive frames, and once a detection is stable enough, a lock
  callback is triggered.
  """
  
  def __init__(
    self,
    confidence_threshold: float,
    overlap_threshold: float,
    confirmation_frames: int,
    disappear_frames: int,
    on_lock_callback: Optional[Callable[[str], None]] = None
  ) -> None:
    """
    Initialize the CardTracker with tracking parameters and an optional callback.
    
    Parameters:
      confidence_threshold (float): Minimum confidence required for a detection to be considered stable.
      overlap_threshold (float): Minimum overlap required to match new detections with existing tracks.
      confirmation_frames (int): Number of consecutive frames required to confirm a detection.
      disappear_frames (int): Maximum allowed missing frames before a track is discarded.
      on_lock_callback (Optional[Callable[[str], None]]): Function to call when a card becomes locked.
    """
    self.confidence_threshold = confidence_threshold
    self.overlap_threshold = overlap_threshold
    self.confirmation_frames = confirmation_frames
    self.disappear_frames = disappear_frames
    self.on_lock_callback = on_lock_callback
    # Dictionary to store currently tracked cards; key is bounding box, value is detection information
    self.tracked_cards: Dict[tuple, Dict[str, Any]] = {}

  def update(self, detections: Dict[tuple, Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    """
    Update tracked card information using new detections.
    
    This method attempts to match each new detection with existing tracked cards by checking the overlap. For
    matched detections, it updates the stability counters and missing frame counts. If a detection becomes
    stable over a specified number of frames, it is marked as locked and triggers a callback. Unmatched tracks
    have their missing frame counts incremented and are removed if they exceed the allowed threshold.
    
    Parameters:
      detections (Dict[tuple, Dict[str, Any]]): A dictionary mapping bounding box coordinates to their
      corresponding detection details.
    
    Returns:
      Dict[tuple, Dict[str, Any]]: A dictionary mapping bounding box coordinates (as tuples) to their tracking
      information (e.g., label, confidence, stable_frames, missing_frames, locked).
    """
    # Create a copy of existing tracked cards for reference
    old_tracked = self.tracked_cards.copy()
    new_tracked: Dict[tuple, Dict[str, Any]] = {}

    # Set to store bounding boxes that have been matched from previous tracking
    matched_prev_keys = set()

    # Iterate over all new detections
    for box, detection_info in detections.items():
      label = detection_info.get("label")
      confidence = detection_info.get("confidence")
      matched_prev = None

      # Attempt to find a matching previously tracked card based on the overlap threshold
      for prev_box, info in old_tracked.items():
        if compute_overlap(list(box), list(prev_box)) >= self.overlap_threshold:
          matched_prev = prev_box
          break

      if matched_prev:
        info = old_tracked[matched_prev]

        # If the detection's confidence exceeds the threshold, increase the stable frame counter
        if confidence >= self.confidence_threshold:
          info["stable_frames"] = info.get("stable_frames", 0) + 1
          info["missing_frames"] = 0
        # Otherwise, increment the missing frame counter
        else:
          info["missing_frames"] = info.get("missing_frames", 0) + 1
        # Once the detection has been stable for enough frames, trigger the lock callback if not already locked
        if info["stable_frames"] >= self.confirmation_frames and not info.get("locked", False):
          if self.on_lock_callback:
            self.on_lock_callback(info["label"])

          info["locked"] = True

        # Update tracking with the new bounding box for the matched detection
        new_tracked[box] = info
        matched_prev_keys.add(matched_prev)
      # For new detections, initialize the tracking info
      else:
        new_tracked[box] = {
          "label": label,
          "confidence": confidence,
          "stable_frames": 1,
          "missing_frames": 0,
          "locked": False
        }

    # Process previously tracked cards that did not match any new detections
    for prev_box, info in old_tracked.items():
      if prev_box not in matched_prev_keys:
        # Increment missing frame count for unmatched tracks
        info["missing_frames"] = info.get("missing_frames", 0) + 1

        if info["missing_frames"] < self.disappear_frames:
          new_tracked[prev_box] = info

    # Update the tracked cards with the new tracking state
    self.tracked_cards = new_tracked
    return new_tracked
