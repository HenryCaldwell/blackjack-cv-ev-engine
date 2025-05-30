from typing import Any, Union

import cv2

from psrc.core.interfaces.i_frame_reader import IFrameReader


class CVVideoStream(IFrameReader):
    """
    CVVideoStream is an implementation of the IFrameReader interface.

    This implementation uses OpenCV’s VideoCapture to open a video source, read frames on demand, report the
    source’s FPS, and release resources.
    """

    def __init__(self, source: Union[int, str] = 0) -> None:
        """
        Initialize CVVideoStream with the given source.

        Parameters:
            source (Union[int, str]): The video source to open. This can be a file path or integer.

        Raises:
            IOError: If the video source cannot be opened.
        """
        # Open the video stream using OpenCV VideoCapture with the specified source
        self.stream = cv2.VideoCapture(source)

        if not self.stream.isOpened():
            raise IOError(f"Unable to open video source: {source}")

    def read_frame(self) -> Any:
        """
        Read the next frame from the video source.

        This implementation returns the frame if successful, or None when no frame is available.

        Returns:
            Any: The next frame, or None if no frame is available.
        """
        # Retrieve the next frame from the video stream
        ret, frame = self.stream.read()

        if not ret:
            return None

        return frame

    def get_fps(self) -> float:
        """
        Query the frame rate of the video source.

        This implementation retrieves CAP_PROP_FPS from the VideoCapture. If the property is missing or
        non-positive, it defaults to 30.0 FPS.

        Returns:
            float: The frames per second of the source.
        """
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def release(self) -> None:
        """
        Release the video source and perform cleanup.

        This implementation releases the stream if the capture is open.

        Returns:
            None
        """
        if self.stream is not None:
            self.stream.release()
