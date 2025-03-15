from abc import ABC, abstractmethod
from typing import Any, Dict, List

class IAnnotator(ABC):
  """
  Interface for annotating video frames with detection and hand information.

  This interface defines the contract for annotators that overlay information (e.g., bounding boxes, hand labels,
  scores) on video frames based on detected card regions and associated hand data.
  """

  @abstractmethod
  def annotate(self, frame: Any, raw_boxes: List[List[float]], hand_dict: Dict[str, Any]) -> Any:
    """
    Annotate a video frame with detection boxes and hand details.

    Parameters:
      frame (Any): The image frame to annotate.
      raw_boxes (List[List[float]]): A list of bounding boxes representing detected card regions.
      hand_dict (Dict[str, Any]): A dictionary containing information about each hand (cards, score, etc.).

    Returns:
      Any: The annotated video frame.
    """
    pass