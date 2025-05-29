from abc import ABC, abstractmethod

from typing import Any


class IFrameReader(ABC):
    """
    Interface for reading frames from a video source.

    This interface defines a contract for retrieving frames, querying frame rate, and releasing resources.
    """

    @abstractmethod
    def read_frame(self) -> Any:
        """
        Read the next frame from the video source.

        Returns:
            Any: The next frame, or None if no frame is available.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """
        Query the frame rate of the video source.

        Returns:
            float: The frames per second of the source.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release the video source and perform cleanup.

        Returns:
            None
        """
        pass
