from abc import ABC, abstractmethod
from typing import Any

class IVideoStreamReader(ABC):
  """
  Interface for reading video streams.

  This interface specifies methods for retrieving frames from a video source and releasing the stream.
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
  def release(self) -> None:
    """
    Release the video stream and any associated resources.
    """
    pass
