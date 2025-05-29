from typing import Any, Dict, List, Tuple

import numpy as np

from psrc.core.interfaces.i_hand_tracker import IHandTracker


class HandTracker(IHandTracker):
    """
    HandTracker is an implementation of the IHandTracker interface.

    This implementation groups confirmed card tracks into blackjack hands based on bounding-box overlap,
    scores each hand, and maintains the latest hand state.
    """

    def __init__(self, overlap_threshold: float = 0.1) -> None:
        """
        Initialize HandTracker with an overlap threshold.

        Parameters:
            overlap_threshold (float): Minimum ratio of intersection to smaller-box area for boxes to be
            considered in the same hand.
        """
        self.hands = {}
        self.overlap_threshold = overlap_threshold

    def _score_hand(self, cards: List[int]) -> int:
        """
        Score a hand based on blackjack rules.

        This method counts aces as 1, with an option to add 10 if it does not bust the hand. Cards with values 1
        through 8 are valued at card value + 1. Cards with value 9 and above are counted as 10.

        Parameters:
            cards (List[int]): A list of card labels.

        Returns:
            int: The best hand value less than or equal to 21.
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
        Compute overlap ratios between all pairs of boxes.

        This method calculates intersection area over the smaller box’s area for each pair.

        Parameters:
            boxes (np.ndarray): An array of bounding boxes of shape (N, 4), with each box as [x_min, y_min,
            x_max, y_max].

        Returns:
            np.ndarray: A (N, N) matrix where each element represents the overlap ratio between two boxes.
        """
        # Extract the coordinates for each bounding box
        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        x_max = boxes[:, 2]
        y_max = boxes[:, 3]

        # For each pair, determine the coordinates of the intersection rectangle
        x_left = np.maximum(x_min[:, None], x_min[None, :])
        y_top = np.maximum(y_min[:, None], y_min[None, :])
        x_right = np.minimum(x_max[:, None], x_max[None, :])
        y_bottom = np.minimum(y_max[:, None], y_max[None, :])

        # Compute intersection dimensions, ensuring no negative values
        inter_width = np.maximum(0, x_right - x_left)
        inter_height = np.maximum(0, y_bottom - y_top)
        inter_area = inter_width * inter_height

        area = (x_max - x_min) * (y_max - y_min)  # Calculate area for each bounding box
        min_area = np.minimum(
            area[:, None], area[None, :]
        )  # For each pair, use the smaller area for the overlap ratio

        overlap = inter_area / (
            min_area + 1e-6
        )  # Calculate the overlap ratio for each pair (with epsilon to avoid division by zero)

        return overlap

    def _group_cards(
        self, boxes: List[Tuple[float, float, float, float]]
    ) -> List[List[int]]:
        """
        Group boxes into clusters where overlap is greater than or equal to the overlap threshold.

        This method builds an adjacency matrix from overlap ratios and uses union-find to cluster connected
        indices into hands.

        Parameters:
            boxes (List[Tuple[float, float, float, float]]): A list of bounding boxes.

        Returns:
            List[List[int]]: A list of groups, where the inner list is the indices of boxes in one hand.
        """
        n = len(boxes)

        # Return empty list if no boxes provided
        if n == 0:
            return []

        boxes_np = np.array(boxes).reshape(
            -1, 4
        )  # Convert list of boxes to a NumPy array for efficient computation
        overlap_matrix = self._compute_overlap_matrix(
            boxes_np
        )  # Compute the pairwise overlap matrix between bounding boxes

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
        Update hands using new card tracks.

        This implementation filters to only confirmed tracks, groups their bounding boxes into hands, assigns
        single-box groups to the dealer and multi-box groups to players in left-to-right order, scores each
        hand, and returns the assembled mapping.

        Parameters:
            tracks (Dict[int, Dict[str, Any]]): A mapping of track IDs to their tracking information.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of hand IDs to their hand information.
        """
        # Retrieve only the confirmed cards from the tracked cards
        stable_tracks = {
            tid: info for tid, info in tracks.items() if info.get("state", 1) == 1
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
            dealer_score = self._score_hand(dealer_cards)
            dealer_boxes = [boxes[idx] for idx in dealer_indices]
            hands_info["Dealer"] = {
                "cards": dealer_cards,
                "score": dealer_score,
                "boxes": dealer_boxes,
            }

        # Identify groups with multiple cards
        player_groups = [group for group in groups if len(group) > 1]
        player_groups.sort(
            key=lambda group: min(boxes[idx][0] for idx in group) if group else 0
        )

        # Process each player group and compute the hand information
        for i, group in enumerate(player_groups, start=1):
            player_cards = [labels[idx] for idx in group]
            score = self._score_hand(player_cards)
            hand_boxes = [boxes[idx] for idx in group]
            hands_info[f"Player {i}"] = {
                "cards": player_cards,
                "score": score,
                "boxes": hand_boxes,
            }

        self.hands = hands_info
        return hands_info
