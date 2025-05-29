import cv2

from typing import Any, Union

from psrc.core.interfaces.i_frame_reader import IFrameReader


class CVVideoStream(IFrameReader):
    """
    CVVideoStream implements the IVideoStreamReader interface for reading video streams using OpenCV.

    This class provides methods for retrieving video frames from a specified source and releasing the stream when
    it is no longer needed.
    """

    def __init__(self, source: Union[int, str] = 0) -> None:
        """
        Initialize the CVVideoStream with a video source.

        Parameters:
          source (Union[int, str]): The video source to open. This can be an integer for a webcam or a file path.

        Raises:
          IOError: If the video source cannot be opened.
        """
        # Open the video stream using OpenCV VideoCapture with the specified source
        self.stream = cv2.VideoCapture(source)

        if not self.stream.isOpened():
            raise IOError(f"Unable to open video source: {source}")

    def read_frame(self) -> Any:
        """
        Read and return the next frame from the video stream.

        Returns:
          Any: The next video frame, or None if no frame is available.
        """
        # Retrieve the next frame from the video stream
        ret, frame = self.stream.read()

        if not ret:
            return None

        return frame

    def get_fps(self) -> float:
        """
        Query the source's frame rate.

        Returns:
          float: Frames per second of the video source; defaults to 30.0 if unavailable.
        """
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def release(self) -> None:
        """
        Release the video stream and any associated resources.
        """
        if self.stream is not None:
            self.stream.release()
