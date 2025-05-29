import jpype

from typing import Dict, List

from psrc.core.interfaces.i_ev_calculator import IExpectedValueCalculator
from psrc.evaluation.java_conversion_utils import (
    deck_to_java_array,
    hand_to_java_array_list,
)


class EVCalculatorWrapper(IExpectedValueCalculator):
    """
    EVCalculatorWrapper implements the IEVCalculator interface for calculating expected values (EV) for blackjack
    actions.

    This class acts as a wrapper around a Java-based EV calculator. It starts the JVM if not already running,
    loads the EVCalculator Java class, and delegates EV calculations for actions like "stand", "hit", "double",
    "split", and "surrender". It also provides a method to gracefully shutdown the JVM.
    """

    def __init__(
        self,
        jar_path: str = "target/blackjack-ev-calculator-1.0.0.jar",
        java_class: str = "evaluation.EVCalculator",
    ) -> None:
        """
        Initialize the EVCalculatorWrapper with the path to the EV calculator JAR and the Java class name.

        Parameters:
          jar_path (str): Path to the Java Archive (JAR) containing the EVCalculator implementation.
          java_class (str): The fully qualified name of the Java class for EV calculations.
        """
        self.jar_path = jar_path
        self.java_class = java_class
        # Start the JVM and initialize the Java EV calculator
        self._start_jvm()

    def _start_jvm(self) -> None:
        """
        Start the Java Virtual Machine (JVM) if it is not already running, and initialize the EVCalculator
        instance.

        This method checks if the JVM is started, and if not, it starts the JVM with the specified classpath. Then
        it loads the EVCalculator Java class and creates an instance for further EV calculations.
        """
        # Start JVM if not already running
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[self.jar_path])

        # Load the EVCalculator Java class and create an instance
        self._java_ev_cls = jpype.JClass(self.java_class)
        self._java_ev = self._java_ev_cls()

    def calculate_stand_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Compute the stand EV by delegating to the Java EVCalculator.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the stand decision.
        """
        return float(
            self._java_ev.calculateStandEV(
                deck_to_java_array(deck),
                hand_to_java_array_list(player_hand),
                hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_hit_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Compute the hit EV by delegating to the Java EVCalculator.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the hit decision.
        """
        return float(
            self._java_ev.calculateHitEV(
                deck_to_java_array(deck),
                hand_to_java_array_list(player_hand),
                hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_double_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Compute the double EV by delegating to the Java EVCalculator.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the double decision.
        """
        return float(
            self._java_ev.calculateDoubleEV(
                deck_to_java_array(deck),
                hand_to_java_array_list(player_hand),
                hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_split_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Compute the split EV by delegating to the Java EVCalculator.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the split decision.
        """
        return float(
            self._java_ev.calculateSplitEV(
                deck_to_java_array(deck),
                hand_to_java_array_list(player_hand),
                hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_surrender_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Compute the surrender EV by returning the set value -0.5.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the surrender decision.
        """
        return -0.5

    def release(self) -> None:
        """
        Release resources by shutting down the Java Virtual Machine (JVM) if it is running.

        This method ensures that the JVM is gracefully shut down when it is no longer needed.
        """
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
