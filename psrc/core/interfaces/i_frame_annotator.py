from abc import ABC, abstractmethod

from typing import Any, Dict, Tuple


class IFrameAnnotator(ABC):
    """
    Interface for annotating frames with detection information.

    This interface defines a contract for annotating a frame with bounding boxes, labels, and states.
    """

    @abstractmethod
    def annotate(
        self,
        frame: Any,
        detections: Dict[Tuple, Dict[str, Any]],
        tracks: Dict[int, Dict[str, Any]],
    ) -> Any:
        """
        Annotate a frame with detection and tracking information.

        Parameters:
            frame (Any): The frame to annotate.
            detections (Dict[Tuple, Dict[str, Any]]): A mapping of bounding box coordinates to their detection
            information.
            tracks (Dict[int, Dict[str, Any]]): A mapping of track IDs to their tracking information.

        Returns:
            Any: The annotated frame.
        """
        pass
