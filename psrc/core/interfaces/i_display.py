from abc import ABC, abstractmethod

from typing import Any, Dict


class IDisplay(ABC):
    """
    Interface for rendering frames and managing UI lifecycle.

    This interface defines a contract for rendering frames, processing UI events, updating display state, and
    performing cleanup of all associated resources.
    """

    @abstractmethod
    def update(
        self,
        *,
        frame: Any = None,
        detections: Dict[tuple, Dict[str, Any]] = None,
        tracks: Dict[int, Dict[str, Any]] = None,
        hands: Dict[str, Dict[str, Any]] = None,
        evals: Dict[str, Dict[str, Any]] = None,
        deck: Dict[int, int] = None,
    ) -> None:
        """
        Update any subset of the display state.

        Parameters:
            frame (Any, optional): The frame for display.
            detections (Dict[tuple, Dict[str, Any]], optional): A mapping of bounding box coordinates to their
            detection information.
            tracks (Dict[int, Dict[str, Any]], optional): A mapping of track IDs to their tracking information.
            hands (Dict[str, Dict[str, Any]], optional): A mapping of hand IDs to their hand information.
            evals (Dict[str, Dict[str, Any]], optional): A mapping of hand IDs to their evaluation information.
            deck (Dict[int, int], optional): A mapping of card label to count.

        Returns:
            None
        """
        pass

    @abstractmethod
    def process_events(self) -> bool:
        """
        Process UI events and determine whether the display should continue running.

        Returns:
            bool: True to keep running, False otherwise.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release all display resources and perform cleanup.

        Returns:
            None
        """
        pass
