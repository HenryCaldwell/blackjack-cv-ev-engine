from abc import ABC, abstractmethod
from typing import Any, Dict


class IDisplay(ABC):
    """
    Interface for rendering video frames, updating UI components, and managing display lifecycle.

    This interface defines hooks for frame rendering, tracking updates, hand grouping, EV updates, deck state,
    and resource cleanup.
    """

    @abstractmethod
    def update(
        self,
        *,
        frame: Any = None,
        detections: Dict[tuple, Dict[str, Any]] = None,
        tracks: Dict[int, Dict[str, Any]] = None,
        hands: Dict[str, Any] = None,
        evals: Dict[str, Any] = None,
        deck: Dict[int, int] = None,
    ) -> None:
        """
        Update any subset of the display’s state.

        Parameters:
          frame (Any, optional): The latest video frame to display.
          detections (Dict[tuple, Dict[str, Any]], optional): Raw detection boxes mapped to detection info.
          tracks (Dict[int, Dict[str, Any]], optional): Tracked card IDs mapped to bbox/label/state.
          hands (Dict[str, Any], optional): Grouped hand information.
          evals (Dict[str, Any], optional): Expected value results and best actions for each hand.
          deck (Dict[int, int], optional): Current deck composition as card label count.
        """
        pass

    @abstractmethod
    def process_events(self) -> bool:
        """
        Pump UI events and determine whether the display should continue running.

        Returns:
          bool: True to keep running; False to initiate shutdown.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release all display resources and perform any necessary cleanup.
        """
        pass
