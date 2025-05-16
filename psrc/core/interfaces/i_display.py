from abc import ABC, abstractmethod
from typing import Any, Dict


class IDisplay(ABC):
    """
    Interface for rendering video frames, updating UI components, and managing display lifecycle.

    This interface defines hooks for frame rendering, tracking updates, hand grouping, EV updates, deck state,
    and resource cleanup.
    """

    @abstractmethod
    def update_frame(self, frame: Any) -> None:
        """
        Update the display with the provided frame and render it.

        Parameters:
          frame (Any): The image frame to be displayed.
        """
        pass

    @abstractmethod
    def update_tracking(self, tracks: Dict[int, Dict[str, Any]]) -> None:
        """
        Receive updated tracking information for each detected object.

        Parameters:
          tracks (Dict[int, Dict[str, Any]]): Mapping from track IDs to track details.
        """
        pass

    @abstractmethod
    def update_hands(self, hands_info: Dict[str, Any]) -> None:
        """
        Receive updated grouping of tracks into hands.

        Parameters:
          hands_info (Dict[str, Any]): Mapping from hand IDs to their details.
        """
        pass

    @abstractmethod
    def update_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """
        Receive updated expected value (EV) results for each hand.

        Parameters:
          eval_results (Dict[str, Any]): Mapping from hand identifiers to a structure containing EVs for available
          actions and the recommended best action.
        """
        pass

    @abstractmethod
    def update_deck(self, deck_state: Dict[int, int]) -> None:
        """
        Receive updated card deck composition after confirmations or removals.

        Parameters:
          deck_state (Dict[int, int]): Mapping from card labels to remaining counts in the deck.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release all display resources and perform any necessary cleanup.
        """
        pass
