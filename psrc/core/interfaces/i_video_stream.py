from abc import ABC, abstractmethod
from typing import Any


class IVideoStreamReader(ABC):
    """
    Interface for ingesting frames from a video source.

    This interface specifies methods for retrieving frames from a video source, querying the sources frame rate,
    and releasing resources.
    """

    @abstractmethod
    def read_frame(self) -> Any:
        """
        Read and return the next frame from the video stream.

        Returns:
          Any: The next video frame, or None if no frame is available.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """
        Query the frame rate of the video source.

        Returns:
          float: Frames per second provided by the source.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release the video stream and any associated resources.
        """
        pass
