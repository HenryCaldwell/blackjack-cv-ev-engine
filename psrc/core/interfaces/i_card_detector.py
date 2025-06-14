from abc import ABC, abstractmethod

from typing import Any, Dict, Tuple


class ICardDetector(ABC):
    """
    Interface for detecting cards in a frame.

    This interface defines a contract for detecting cards using computer vision techniques.
    """

    @abstractmethod
    def detect(self, frame: Any) -> Dict[Tuple, Dict[str, Any]]:
        """
        Detect cards within a given frame.

        Parameters:
            frame (Any): The frame in which to detect.

        Returns:
            Dict[Tuple, Dict[str, Any]]: A mapping of bounding box coordinates to their detection information.
        """
        pass
