import numpy as np

from typing import Any, Dict, List, Tuple

from psrc.core.interfaces.i_hand_tracker import IHandTracker

class HandTracker(IHandTracker):
  """
  HandTracker implements the IHandTracker interface for tracking and grouping card tracks into blackjack hands.

  This class groups card tracks into hands using an overlap-based grouping algorithm and evaluates the hand
  scores using a predefined evaluation method. Hands with a single card are assumed to belong to the dealer,
  while hands with multiple cards are assumed to belong to players.
  """

  def __init__(self, overlap_threshold: float = 0.1) -> None:
    """
    Initialize the HandTracker with an empty state and set the overlap threshold.

    Parameters:
      overlap_threshold (float): Minimum required overlap between bounding boxes to consider them part of the
      same group.
    """
    self.hands_state = {}  # Dictionary to store current hands with hand index and dealer as keys
    self.overlap_threshold = overlap_threshold

  def _evaluate_hand(self, cards: List[int]) -> int:
    """
    Evaluate the total score of a hand based on card values.

    Ace (represented by 0) is initially counted as 1, with an option to add 10 if it does not bust the hand.
    Cards with values 1 through 8 are valued at card value + 1. Cards with value 9 and above are counted as 10.

    Parameters:
      cards (List[int]): A list of card labels.

    Returns:
      int: The calculated score of the hand.
    """
    total = 0
    aces = 0

    # Iterate through each card in the hand and update total and ace count accordingly
    for card in cards:
      if card == 0:
        total += 1
        aces += 1
      elif 1 <= card <= 8:
        total += card + 1
      else:
        total += 10

    # If possible, adjust Aces by adding 10 without busting
    while aces > 0 and total + 10 <= 21:
      total += 10
      aces -= 1

    return int(total)

  def _compute_overlap_matrix(self, boxes: np.ndarray) -> np.ndarray:
    """
    Compute the overlap matrix for a set of bounding boxes.

    The overlap for a pair of boxes is computed as the ratio of the area of their intersection to the area of
    the smaller box. This results in a symmetric matrix where each element (i, j) indicates the overlap between
    boxes i and j.

    Parameters:
      boxes (np.ndarray): An array of bounding boxes of shape (N, 4), where each box is represented by [x_min,
      y_min, x_max, y_max].

    Returns:
      np.ndarray: A (N, N) matrix where each element represents the overlap ratio between two boxes.
    """
    # Extract the coordinates for each bounding box
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    
    # For each pair, determine the coordinates of the intersection rectangle
    x_left   = np.maximum(x_min[:, None], x_min[None, :])
    y_top    = np.maximum(y_min[:, None], y_min[None, :])
    x_right  = np.minimum(x_max[:, None], x_max[None, :])
    y_bottom = np.minimum(y_max[:, None], y_max[None, :])
    
    # Compute intersection dimensions, ensuring no negative values
    inter_width  = np.maximum(0, x_right - x_left)
    inter_height = np.maximum(0, y_bottom - y_top)
    inter_area   = inter_width * inter_height
    
    area = (x_max - x_min) * (y_max - y_min)  # Calculate area for each bounding box
    min_area = np.minimum(area[:, None], area[None, :])  # For each pair, use the smaller area for the overlap ratio
    
    overlap = inter_area / (min_area + 1e-6)  # Calculate the overlap ratio for each pair (with epsilon to avoid division by zero)

    return overlap

  def _group_cards(self, boxes: List[Tuple[float, float, float, float]]) -> List[List[int]]:
    """
    Group cards based on the overlap of their bounding boxes using a union-find algorithm.

    The method computes an overlap matrix and considers two boxes as connected if their overlap exceeds the
    defined threshold. It then uses a union-find (disjoint set) data structure to cluster connected boxes
    together.

    Parameters:
      boxes (List[Tuple[float, float, float, float]]): List of bounding boxes for the detected cards.

    Returns:
      List[List[int]]: A list of groups, where each group is a list of indices corresponding to boxes that
      belong to the same hand.
    """
    n = len(boxes)
    
    # Return empty list if no boxes provided
    if n == 0:
      return []

    boxes_np = np.array(boxes).reshape(-1, 4)  # Convert list of boxes to a NumPy array for efficient computation
    overlap_matrix = self._compute_overlap_matrix(boxes_np)  # Compute the pairwise overlap matrix between bounding boxes
    
    # Create an adjacency matrix where it is True if overlap is above the threshold
    adj = overlap_matrix >= self.overlap_threshold
    np.fill_diagonal(adj, False)
    
    parent = list(range(n))
    
    # Find the representative of the set containing x using path compression
    def _find(x: int) -> int:
      while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
      return x
    
    # Merge the sets containing x and y
    def _union(x: int, y: int) -> None:
      root_x = _find(x)
      root_y = _find(y)

      if root_x != root_y:
        parent[root_y] = root_x
    
    # Iterate over all pairs of boxes and union their sets if they are adjacent
    for i in range(n):
      for j in range(i + 1, n):
        if adj[i, j]:
          _union(i, j)
    
    # Group boxes based on their representative parent
    groups_dict: Dict[int, List[int]] = {}

    for i in range(n):
      root = _find(i)
      
      if root not in groups_dict:
        groups_dict[root] = []

      groups_dict[root].append(i)
    
    return list(groups_dict.values())

  def update(self, tracks: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update the hand tracker with new card tracks and group them into hands.

    This method takes in a dictionary of card tracks, groups the associated bounding boxes into hands using an
    overlap-based grouping algorithm, and evaluates the score for each hand. A single-card group is assumed to
    be the dealer's hand, while groups with multiple cards are considered player hands.

    Parameters:
      tracks (Dict[int, Dict[str, Any]]): Dictionary of card tracks, where each track contains detection info
      such as 'bbox' and 'label'.

    Returns:
      Dict[str, Any]: A dictionary mapping hand identifiers to their hand details (e.g., cards, score, boxes).
    """
    # Retrieve only the confirmed cards from the tracked cards
    stable_tracks = {
      tid: info for tid, info in tracks.items()
      if info.get("state", 1) == 1
    }
    
    boxes = [tuple(info["bbox"]) for info in stable_tracks.values()]
    labels = [info["label"] for info in stable_tracks.values()]
    
    # Group the bounding boxes based on their overlap
    groups = self._group_cards(boxes)
    hands_info: Dict[str, Any] = {}
    
    # Identify groups that consist of a single card
    single_groups = [group for group in groups if len(group) == 1]
    dealer_indices = [idx for group in single_groups for idx in group]

    # Extract dealer card labels and boxes from the indices
    if dealer_indices:
      dealer_cards = [labels[idx] for idx in dealer_indices]
      dealer_score = self._evaluate_hand(dealer_cards)
      dealer_boxes = [boxes[idx] for idx in dealer_indices]
      hands_info["Dealer"] = {"cards": dealer_cards, "score": dealer_score, "boxes": dealer_boxes}
    
    # Identify groups with multiple cards
    player_groups = [group for group in groups if len(group) > 1]
    player_groups.sort(key=lambda group: min(boxes[idx][0] for idx in group) if group else 0)

    # Process each player group and compute the hand information
    for i, group in enumerate(player_groups, start=1):
      player_cards = [labels[idx] for idx in group]
      score = self._evaluate_hand(player_cards)
      hand_boxes = [boxes[idx] for idx in group]
      hands_info[f"Player {i}"] = {"cards": player_cards, "score": score, "boxes": hand_boxes}
    
    self.hands_state = hands_info
    return hands_info