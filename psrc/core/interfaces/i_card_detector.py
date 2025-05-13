from abc import ABC, abstractmethod
from typing import Any, Dict


class ICardDetector(ABC):
    """
    Interface for detecting cards in a video frame.

    This interface provides a contract for detecting cards using computer vision techniques, returning bounding
    boxes along with associated detection details (like label and confidence).
    """

    @abstractmethod
    def detect(self, frame: Any) -> Dict[tuple, Dict[str, Any]]:
        """
        Detect cards within a given frame.

        Parameters:
          frame (Any): The image frame in which to detect cards.

        Returns:
          Dict[tuple, Dict[str, Any]]: A dictionary mapping bounding box coordinates (as tuples) to their
          detection information (e.g., label, confidence).
        """
        pass
