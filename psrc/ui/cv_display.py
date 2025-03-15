import cv2

from typing import Any

from psrc.core.interfaces.i_display import IDisplay

class CVDisplay(IDisplay):
  """
  CVDisplay implements the IDisplay interface for displaying video frames using OpenCV.

  This class creates an OpenCV window to render video frames, handles user input (such as quit commands),
  and cleans up display resources when no longer needed.
  """
  
  def __init__(self, window_name: str = "Vision Display") -> None:
    """
    Initialize the CVDisplay with a specified window name.
    
    Parameters:
      window_name (str): The name of the display window.
    """
    self.window_name = window_name
  
  def update(self, frame: Any) -> None:
    """
    Update the display with the provided frame and render it.
    
    Parameters:
      frame (Any): The image frame to be displayed.
    """
    cv2.imshow(self.window_name, frame)
  
  def handle_input(self) -> bool:
    """
    Handle user input events for the display.
    
    Returns:
      bool: True if processing should continue, False if an exit command (e.g., 'q') is received.
    """
    # Wait for a key event for 1 ms and capture the key code
    key = cv2.waitKey(1) & 0xFF
    
    # Check if the 'q' key was pressed to signal exit
    if key == ord('q'):
      return False
    
    return True
  
  def release(self) -> None:
    """
    Release the display and any associated resources.
    """
    cv2.destroyAllWindows()