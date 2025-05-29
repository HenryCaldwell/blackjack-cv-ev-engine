from abc import ABC, abstractmethod

from typing import Any, Dict, Tuple


class ICardTracker(ABC):
    """
    Interface for tracking card detections across frames.

    This interface defines a contract for updating tracking information based on new detections.
    """

    @abstractmethod
    def update(
        self, detections: Dict[Tuple, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Update tracked cards using new detections.

        Parameters:
            detections (Dict[Tuple, Dict[str, Any]]): A mapping of bounding box coordinates to their detection
            information.

        Returns:
            Dict[int, Dict[str, Any]]: A mapping of track IDs to their tracking information.
        """
        pass
