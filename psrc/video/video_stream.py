import cv2

from typing import Any, Union

from psrc.core.interfaces.i_video_stream import IVideoStreamReader

class VideoStreamReader(IVideoStreamReader):
  def __init__(self, source: Union[int, str] = 0) -> None:
    self.stream = cv2.VideoCapture(source)

    if not self.stream.isOpened():
      error_msg = f"Unable to open video source: {source}"
      raise IOError(error_msg)
  
  def read_frame(self) -> Any:
    ret, frame = self.stream.read()

    if not ret:
      return None
    
    return frame
  
  def release(self) -> None:
    if self.stream is not None:
      self.stream.release()