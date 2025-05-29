from abc import ABC, abstractmethod

from typing import Any, Dict


class IHandTracker(ABC):
    """
    Interface for grouping tracked cards into blackjack hands.

    This interface defines a contract for updating hand information based on tracking information.
    """

    @abstractmethod
    def update(self, tracks: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Update hands using new card tracks.

        Parameters:
            tracks (Dict[int, Dict[str, Any]]): A mapping of track IDs to their tracking information.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of hand IDs to their hand information.
        """
        pass
