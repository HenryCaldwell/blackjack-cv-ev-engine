from typing import Any, Dict, List

import jpype

from psrc.core.interfaces.i_ev_calculator import IExpectedValueCalculator


class EVCalculatorWrapper(IExpectedValueCalculator):
    """
    EVCalculatorWrapper is an implementation of the IExpectedValueCalculator interface.

    This implementation starts a JVM, wraps a Java EVCalculator class, and forwards expected-value computation
    calls by converting Python data structures into Java arrays/lists.
    """

    def __init__(
        self,
        jar_path: str = "target/blackjack-ev-calculator-1.0.0.jar",
        java_class: str = "evaluation.EVCalculator",
    ) -> None:
        """
        Initialize EVCalculatorWrapper by launching the JVM and instantiating the Java calculator.

        Parameters:
            jar_path (str): The path to the EV calculator JAR.
            java_class (str): The fully qualified Java class name.
        """
        self.jar_path = jar_path
        self.java_class = java_class
        # Start the JVM and initialize the Java EV calculator
        self._start_jvm()

    def _start_jvm(self) -> None:
        """
        Ensure the JVM is running and create a Java EVCalculator instance.

        This method checks jpype.isJVMStarted(), starts it if needed using the provided JAR, and then
        loads and instantiates the Java class.
        """
        # Start JVM if not already running
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[self.jar_path])

        # Load the EVCalculator Java class and create an instance
        self._java_ev_cls = jpype.JClass(self.java_class)
        self._java_ev = self._java_ev_cls()

    def _deck_to_java_array(self, deck: Dict[int, int]) -> Any:
        """
        Convert a Python deck dictionary to a Java array of integers.

        This method orders card counts from label 0 through 9, constructs a Python list of those counts, and uses
        JPype’s JArray and JInt to build a Java integer array.

        Parameters:
            deck (Dict[int, int]): A mapping of card labels to remaining counts.

        Returns:
            Any: A Java integer array containing counts in label order.
        """
        order = list(range(0, 10))
        deck_values = [deck.get(i, 0) for i in order]
        # Convert the Python list to a Java integer array using JArray and JInt
        return jpype.JArray(jpype.JInt)([jpype.JInt(val) for val in deck_values])

    def _hand_to_java_array_list(self, hand: List[int]) -> Any:
        """
        Convert a Python hand list to a Java ArrayList of integers.

        This method normalizes each card label to its blackjack value, wraps each in JInt, and adds to a Java
        ArrayList.

        Parameters:
            hand (List[int]): A list of card labels in the hand.

        Returns:
            Any: A Java integer ArrayList containing normalized card values.
        """
        # Get the Java ArrayList class and create an instance
        ArrayList = jpype.JClass("java.util.ArrayList")
        java_list = ArrayList()

        # Iterate through each card in the hand and normalize its value
        for card in hand:
            if card == 0:
                value = 1
            elif 1 <= card <= 8:
                value = card + 1
            else:
                value = 10

            # Add the normalized value to the Java ArrayList as a JInt
            java_list.add(jpype.JInt(value))

        return java_list

    def calculate_stand_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Calculate the expected value when the player stands.

        This implementation converts the deck and hands to Java arrays/lists and calls the Java method
        calculateStandEV.

        Parameters:
            deck (Dict[int, int]): A mapping of card labels to remaining counts.
            player_hand (List[int]): A list of card labels in the player's hand.
            dealer_hand (List[int]): A list of card labels in the dealer's hand.

        Returns:
            float: The expected value for the stand decision.
        """
        return float(
            self._java_ev.calculateStandEV(
                self._deck_to_java_array(deck),
                self._hand_to_java_array_list(player_hand),
                self._hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_hit_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Calculate the expected value when the player hits.

        This implementation converts the deck and hands to Java arrays/lists and calls the Java method
        calculateHitEV.

        Parameters:
            deck (Dict[int, int]): A mapping of card labels to remaining counts.
            player_hand (List[int]): A list of card labels in the player's hand.
            dealer_hand (List[int]): A list of card labels in the dealer's hand.

        Returns:
            float: The expected value for the hit decision.
        """
        return float(
            self._java_ev.calculateHitEV(
                self._deck_to_java_array(deck),
                self._hand_to_java_array_list(player_hand),
                self._hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_double_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Calculate the expected value when the player doubles.

        This implementation converts the deck and hands to Java arrays/lists and calls the Java method
        calculateDoubleEV.

        Parameters:
            deck (Dict[int, int]): A mapping of card labels to remaining counts.
            player_hand (List[int]): A list of card labels in the player's hand.
            dealer_hand (List[int]): A list of card labels in the dealer's hand.

        Returns:
            float: The expected value for the double decision.
        """
        return float(
            self._java_ev.calculateDoubleEV(
                self._deck_to_java_array(deck),
                self._hand_to_java_array_list(player_hand),
                self._hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_split_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Calculate the expected value when the player splits.

        This implementation converts the deck and hands to Java arrays/lists and calls the Java method
        calculateSplitEV.

        Parameters:
            deck (Dict[int, int]): A mapping of card labels to remaining counts.
            player_hand (List[int]): A list of card labels in the player's hand.
            dealer_hand (List[int]): A list of card labels in the dealer's hand.

        Returns:
            float: The expected value for the split decision.
        """
        return float(
            self._java_ev.calculateSplitEV(
                self._deck_to_java_array(deck),
                self._hand_to_java_array_list(player_hand),
                self._hand_to_java_array_list(dealer_hand),
            )
        )

    def calculate_surrender_ev(
        self,
        deck: Dict[int, int],
        player_hand: List[int],
        dealer_hand: List[int],
    ) -> float:
        """
        Calculate the expected value when the player surrenders.

        This implementation returns a fixed value of -0.5 (no Java call).

        Parameters:
            deck (Dict[int, int]): A mapping of card labels to remaining counts.
            player_hand (List[int]): A list of card labels in the player's hand.
            dealer_hand (List[int]): A list of card labels in the dealer's hand.

        Returns:
            float: The expected value for the surrender decision.
        """
        return -0.5

    def release(self) -> None:
        """
        Shutdown the JVM if it is running.

        This method calls jpype.shutdownJVM to cleanly stop the Java VM.
        """
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
