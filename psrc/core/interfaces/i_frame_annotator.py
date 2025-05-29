import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class IFrameAnnotator(ABC):
    """
    Interface for annotating video frames with detection information.

    This interface defines the contract for annotators that overlay information (e.g., bbox, label, state) on
    video frames based on detected card regions and associated data.
    """

    @abstractmethod
    def annotate(
        self,
        frame: Any,
        detections: Dict[Tuple[int, int, int, int], Dict[str, Any]],
        tracks: Dict[int, Dict[str, Any]],
    ) -> Any:
        """
        Annotate a video frame with detection and track data.

        Parameters:
          frame (Any): The image frame to annotate.
          detections (Dict[Tuple[int, int, int, int], Dict[str, Any]]): A dictionary mapping detection boxes to
          detection metadata.
          tracks (Dict[int, Dict[str, Any]]): A dictionary mapping track numbers to card information
          (e.g., bbox, label, state).

        Returns:
          Any: The annotated video frame.
        """
        pass
