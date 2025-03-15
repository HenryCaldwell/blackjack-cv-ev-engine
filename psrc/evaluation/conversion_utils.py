from typing import Any, Dict, List
from jpype import JInt, JArray, JClass

def deck_to_java_array(deck: Dict[int, int]) -> Any:
  order = list(range(0, 10))
  deck_values = [deck.get(i, 0) for i in order]
  return JArray(JInt)([JInt(val) for val in deck_values])

def hand_to_java_array_list(hand: List[int]) -> Any:
  ArrayList = JClass("java.util.ArrayList")
  java_list = ArrayList()

  for card in hand:
    if card == 0:
      value = 1
    elif 1 <= card <= 8:
      value = card + 1
    else:
      value = 10

    java_list.add(JInt(value))

  return java_list
