import cv2
import logging

from typing import Any, Dict, List
from psrc.core.interfaces.i_annotator import IAnnotator

class CVAnnotator(IAnnotator):
  """
    CVAnnotator implements the IAnnotator interface for annotating video frames with detection and hand
    information.

    This class overlays bounding boxes and hand details on video frames using OpenCV. It uses configurable
    colors, font scales, and thicknesses to differentiate between player and dealer hands.
    """
  
  def __init__(self,
                player_color: tuple = (255, 255, 255),
                dealer_color: tuple = (0, 255, 0),
                default_color: tuple = (100, 100, 100),
                font_scale: float = 0.5,
                thickness: int = 2) -> None:
    """
    Initialize the CVAnnotator with visual styling options.
    
    Parameters:
      player_color (tuple): Color for annotating player hands.
      dealer_color (tuple): Color for annotating the dealer hand.
      default_color (tuple): Default color for annotating other regions.
      font_scale (float): Scale factor for the annotation text.
      thickness (int): Thickness for drawing rectangles and text.
    """
    self.player_color = player_color
    self.dealer_color = dealer_color
    self.default_color = default_color
    self.font_scale = font_scale
    self.thickness = thickness

  def annotate(self, frame: Any, raw_boxes: List[List[float]], hand_dict: Dict[str, Any]) -> Any:
    """
    Annotate a video frame with detection boxes and hand details.
    
    This method overlays bounding boxes and text on the provided frame based on the given hand information. It
    determines the appropriate color for each hand (dealer, player, or default), draws rectangles around
    detected card regions, and places a label indicating hand details (cards and score).
    
    Parameters:
      frame (Any): The image frame to annotate.
      raw_boxes (List[List[float]]): A list of bounding boxes representing detected card regions.
      hand_dict (Dict[str, Any]): A dictionary mapping hand identifiers to their hand details (e.g., cards,
      score, boxes).
    
    Returns:
      Any: The annotated video frame.
    """
    # Process each hand and its associated information
    for hand_name, info in hand_dict.items():
      boxes = info.get("boxes", [])
      cards = info.get("cards", [])
      score = info.get("score", 0)
      
      # Select the annotation color based on hand identifier
      if hand_name.lower() == "dealer":
        color = self.dealer_color
      elif hand_name.lower().startswith("player"):
        color = self.player_color
      else:
        color = self.default_color
      
      # Annotate each bounding box that belongs to the hand
      for box in boxes:
        if box in raw_boxes:
          x1, y1, x2, y2 = map(int, box)
          cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=self.thickness)
          text = f"{hand_name} - Cards: {cards}, Score: {score}"
          cv2.putText(frame, text, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_PLAIN, self.font_scale, color, self.thickness)
    return frame