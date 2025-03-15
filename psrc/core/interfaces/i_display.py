from abc import ABC, abstractmethod
from typing import Any

class IDisplay(ABC):
  """
  Interface for displaying video frames and managing user input.

  This interface specifies the methods needed to update the display with new frames, handle user input (such as
  quit commands), and properly close the display.
  """

  @abstractmethod
  def update(self, frame: Any) -> None:
    """
    Update the display with the provided frame and render it.

    Parameters:
      frame (Any): The image frame to be displayed.
    """
    pass

  @abstractmethod
  def handle_input(self) -> bool:
    """
    Handle user input events for the display.

    Returns:
      bool: True if processing should continue, False if an exit command is received.
    """
    pass

  @abstractmethod
  def release(self) -> None:
    """
    Release the display and any associated resources.
    """
    pass