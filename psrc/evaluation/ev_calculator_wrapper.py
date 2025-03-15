from typing import Dict, List
import jpype
import jpype.imports

from psrc.core.interfaces.i_ev_calculator import IEVCalculator
from psrc.evaluation.java_conversion_utils import deck_to_java_array, hand_to_java_array_list

class EVCalculatorWrapper(IEVCalculator):
  """
  EVCalculatorWrapper implements the IEVCalculator interface for calculating expected values (EV) for blackjack
  actions.

  This class acts as a wrapper around a Java-based EV calculator. It starts the JVM if not already running, 
  loads the EVCalculator Java class, and delegates EV calculations for actions like "stand", "hit", "double", 
  and "split". It also provides a method to gracefully shutdown the JVM.
  """

  def __init__(self, jar_path: str = "target/blackjack-ev-calculator-1.0.0.jar",
                java_class: str = "evaluation.EVCalculator") -> None:
    """
    Initialize the EVCalculatorWrapper with the path to the EV calculator JAR and the Java class name.

    Parameters:
      jar_path (str): Path to the Java Archive (JAR) containing the EVCalculator implementation.
      java_class (str): The fully qualified name of the Java class for EV calculations.
    """
    self.jar_path = jar_path
    self.java_class = java_class
    self.started = False
    # Start the JVM and initialize the Java EV calculator
    self._start_jvm()

  def _start_jvm(self) -> None:
    """
    Start the Java Virtual Machine (JVM) if it is not already running, and initialize the EVCalculator instance.
    
    This method checks if the JVM is started, and if not, it starts the JVM with the specified classpath. Then
    it loads the EVCalculator Java class and creates an instance for further EV calculations.
    """
    # Start JVM if not already running
    if not jpype.isJVMStarted():
      jpype.startJVM(classpath=[self.jar_path])

    # Load the EVCalculator Java class and create an instance
    self.EVCalculatorClass = jpype.JClass(self.java_class)
    self.ev_calculator = self.EVCalculatorClass()

    self.started = True

  def calculate_ev(self, action: str, deck: Dict[int, int],
                    player_hand: List[int], dealer_hand: List[int]) -> float:
    """
    Calculate the expected value (EV) for a given blackjack action.

    This method maps the specified action to the corresponding Java method for EV calculation. It converts the
    deck and hands into Java-compatible types using conversion utilities, then calls the EV calculation method
    for the action.

    Parameters:
      action (str): The blackjack action to evaluate (e.g., "stand", "hit", "double", "split").
      deck (Dict[int, int]): The current deck state as a mapping from card labels to counts.
      player_hand (List[int]): The list of card labels in the player's hand.
      dealer_hand (List[int]): The list of card labels in the dealer's hand.

    Returns:
      float: The computed expected value for the specified action.
    """
    # Map action strings to the corresponding Java method
    method_mapping = {
      "stand": self.ev_calculator.calculateStandEV,
      "hit": self.ev_calculator.calculateHitEV,
      "double": self.ev_calculator.calculateDoubleEV,
      "split": self.ev_calculator.calculateSplitEV,
    }

    # Raise an error if an unknown action is provided
    if action not in method_mapping:
      raise ValueError(f"Unknown action: {action}")

    # Convert Python deck and hand structures to Java-compatible types
    value_counts_java = deck_to_java_array(deck)
    player_hand_java = hand_to_java_array_list(player_hand)
    dealer_hand_java = hand_to_java_array_list(dealer_hand)

    # Call the appropriate Java method to calculate the EV and convert the result to float
    ev = method_mapping[action](value_counts_java, player_hand_java, dealer_hand_java)
    return float(ev)

  def release(self) -> None:
    """
    Release resources by shutting down the Java Virtual Machine (JVM) if it is running.

    This method ensures that the JVM is gracefully shut down when it is no longer needed.
    """
    if jpype.isJVMStarted():
      jpype.shutdownJVM()