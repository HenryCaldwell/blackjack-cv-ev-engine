import cv2
import logging

from typing import Any, Dict, List
from psrc.core.interfaces.i_annotator import IAnnotator

class Annotator(IAnnotator):
  def __init__(self,
                player_color: tuple = (255, 255, 255),
                dealer_color: tuple = (0, 255, 0),
                default_color: tuple = (100, 100, 100),
                font_scale: float = 0.5,
                thickness: int = 2) -> None:
    self.player_color = player_color
    self.dealer_color = dealer_color
    self.default_color = default_color
    self.font_scale = font_scale
    self.thickness = thickness
    self.logger = logging.getLogger(__name__)

  def annotate(self, frame: Any, raw_boxes: List[List[float]], hand_dict: Dict[str, Any]) -> Any:
    for hand_name, info in hand_dict.items():
      boxes = info.get("boxes", [])
      cards = info.get("cards", [])
      score = info.get("score", 0)
      
      if hand_name.lower() == "dealer":
        color = self.dealer_color
      elif hand_name.lower().startswith("player"):
        color = self.player_color
      else:
        color = self.default_color
      
      for box in boxes:
        if box in raw_boxes:
          x1, y1, x2, y2 = map(int, box)
          cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=self.thickness)
          text = f"{hand_name} - Cards: {cards}, Score: {score}"
          cv2.putText(frame, text, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_PLAIN, self.font_scale, color, self.thickness)
    return frame