from typing import Dict, List
import jpype
import jpype.imports

from psrc.core.interfaces.i_ev_calculator import IEVCalculator
from psrc.evaluation.conversion_utils import deck_to_java_array, hand_to_java_array_list

class EVCalculatorWrapper(IEVCalculator):
  def __init__(self, jar_path: str = "target/blackjack-ev-calculator-1.0.0.jar",
                java_class: str = "evaluation.EVCalculator") -> None:
    self.jar_path = jar_path
    self.java_class = java_class
    self.started = False
    self._start_jvm()

  def _start_jvm(self) -> None:
    if not jpype.isJVMStarted():
      jpype.startJVM(classpath=[self.jar_path])

    self.EVCalculatorClass = jpype.JClass(self.java_class)
    self.ev_calculator = self.EVCalculatorClass()
    self.started = True

  def calculate_ev(self, action: str, deck: Dict[int, int],
                    player_hand: List[int], dealer_hand: List[int]) -> float:
    method_mapping = {
      "stand": self.ev_calculator.calculateStandEV,
      "hit": self.ev_calculator.calculateHitEV,
      "double": self.ev_calculator.calculateDoubleEV,
      "split": self.ev_calculator.calculateSplitEV,
    }

    if action not in method_mapping:
      raise ValueError(f"Unknown action: {action}")

    value_counts_java = deck_to_java_array(deck)
    player_hand_java = hand_to_java_array_list(player_hand)
    dealer_hand_java = hand_to_java_array_list(dealer_hand)

    ev = method_mapping[action](value_counts_java, player_hand_java, dealer_hand_java)
    return float(ev)

  def release(self) -> None:
    if jpype.isJVMStarted():
      jpype.shutdownJVM()