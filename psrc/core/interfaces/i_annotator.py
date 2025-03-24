from abc import ABC, abstractmethod
from typing import Any, Dict, List

class IAnnotator(ABC):
  """
  Interface for annotating video frames with detection information.

  This interface defines the contract for annotators that overlay information (e.g., bbox, label, state) on
  video frames based on detected card regions and associated data.
  """

  @abstractmethod
  def annotate(self, frame: Any, raw_boxes: List[List[float]], tracked_cards: Dict[int, Dict[str, Any]]) -> Any:
    """
    Annotate a video frame with tracked card information.

    Parameters:
      frame (Any): The image frame to annotate.
      raw_boxes (List[List[float]]): A list of bounding boxes representing detected card regions.
      tracked_cards (Dict[int, Dict[str, Any]]): A dictionary mapping track numbers to card information (e.g.,
      bbox, label, state).

    Returns:
      Any: The annotated video frame.
    """
    pass