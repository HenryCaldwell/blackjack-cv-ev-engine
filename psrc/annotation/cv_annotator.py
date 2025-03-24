import cv2
import logging

from typing import Any, Dict, List
from psrc.core.interfaces.i_annotator import IAnnotator

class CVAnnotator(IAnnotator):
  """
  CVAnnotator implements the IAnnotator interface for annotating video frames with tracked card detections and
  information.

  This class overlays bounding boxes and detection details on video frames using OpenCV. It uses configurable
  colors, font scales, and thicknesses to differentiate between confirmed and tentative cards.
  """
  
  def __init__(self,
                confirmed_color: tuple = (0, 255, 0),
                tentative_color: tuple = (0, 0, 255),
                font_scale: float = 0.5,
                thickness: int = 2) -> None:
    """
    Initialize the CVAnnotator with visual styling options.
    
    Parameters:
      confirmed_color (tuple): Color for annotating confirmed cards.
      tentative_color (tuple): Color for annotating unconfirmed cards.
      font_scale (float): Scale factor for the annotation text.
      thickness (int): Thickness for drawing rectangles and text.
    """
    self.confirmed_color = confirmed_color
    self.tentative_color = tentative_color
    self.font_scale = font_scale
    self.thickness = thickness

  def annotate(self, frame: Any, raw_boxes: List[List[float]], tracked_cards: Dict[int, Dict[str, Any]]) -> Any:
    """
    Annotate a video frame with tracked card detections.
    
    This method overlays bounding boxes and labels on the provided frame based on the tracked cards dictionary.
    Cards that are not yet confirmed are outlined in red, while confirmed cards are outlined in green.
    
    Parameters:
      frame (Any): The image frame to annotate.
      raw_boxes (List[List[float]]): A list of bounding boxes representing detected card regions.
      tracked_cards (Dict[int, Dict[str, Any]]): A dictionary mapping track numbers to card information (e.g.,
      bbox, label, state).
    
    Returns:
      Any: The annotated video frame.
    """
    # Process each tracked card and its associated information
    for track_id, info in tracked_cards.items():
      bbox = info.get("bbox")

      # Only annotate if the bounding box is present in the raw_boxes list
      if tuple(bbox) not in raw_boxes:
        continue

      label = info.get("label", "N/A")
      state = info.get("state", 0)
      
      # Select the annotation color based on confirmation state
      color = self.confirmed_color if state == 1 else self.tentative_color
      
      x1, y1, x2, y2 = map(int, bbox)
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=self.thickness)
      text = f"ID: {track_id}, Label: {label}"
      cv2.putText(frame, text, (x1, y1 - 10),
                  cv2.FONT_HERSHEY_PLAIN, self.font_scale, color, self.thickness)
    return frame