import cv2

from typing import Any

from psrc.core.interfaces.i_display import IDisplay

class CVDisplay(IDisplay):
  def __init__(self, window_name: str = "Vision Display") -> None:
    self.window_name = window_name
    cv2.namedWindow(self.window_name)
  
  def update_and_show(self, frame: Any) -> None:
    cv2.imshow(self.window_name, frame)
  
  def handle_input(self) -> bool:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
      return False
    
    return True
  
  def release(self) -> None:
    cv2.destroyAllWindows()