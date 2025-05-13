from abc import ABC, abstractmethod
from typing import Any, Dict


class IHandTracker(ABC):
    """
    Interface for tracking and grouping cards into blackjack hands.

    This interface defines a method for updating hand information by grouping detected cards and calculating the
    corresponding hand scores.
    """

    @abstractmethod
    def update(self, detections: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update the hand tracker with new card detections and group them into hands.

        Parameters:
          detections (Dict[int, Dict[str, Any]]): A dictionary mapping track IDs to detection details.

        Returns:
          Dict[str, Any]: A dictionary mapping hand identifiers (as strings) to their hand details (e.g., cards,
          score, boxes).
        """
        pass
