from abc import ABC, abstractmethod
from typing import Any, Dict


class ICardTracker(ABC):
    """
    Interface for tracking card detections across frames.

    This interface defines a method to update the tracking information based on new detections, ensuring stability
    and continuity in card identification.
    """

    @abstractmethod
    def update(
        self, detections: Dict[tuple, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Update the tracked cards using new detections.

        Parameters:
          detections (Dict[tuple, Dict[str, Any]]): A dictionary mapping bounding box coordinates to their
          corresponding detection details.

        Returns:
          Dict[tuple, Dict[str, Any]]: A dictionary mapping track IDs to their tracking information (e.g.,
          bounding box, label, state).
        """
        pass
