from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from psrc.core.interfaces.i_card_tracker import ICardTracker


class TrackState:
    """
    Enumeration for track states.

    Attributes:
      TENTATIVE (int): Newly created track which is not yet confirmed.
      CONFIRMED (int): Track confirmed after consecutive hits.
      DELETED (int): Track marked for deletion after too many misses.
    """

    TENTATIVE = 0
    CONFIRMED = 1
    DELETED = 2


class Track:
    """
    Represents an individual tracked card with its properties.

    Attributes:
        track_id (int): A unique identifier for the track.
        bbox (Tuple[float, float, float, float]): The bounding box coordinates of the card.
        state (int): The current state of the track.
        hits (int): The number of consecutive frames the card was detected.
        misses (int): The number of consecutive frames the card was not detected.
        label (Any): The label associated with the detection.
    """

    def __init__(
        self, track_id: int, bbox: Tuple[float, float, float, float], label: Any
    ) -> None:
        """
        Initialize Track with track parameters.

        Parameters:
            track_id (int): A unique identifier for the track.
            bbox (Tuple[float, float, float, float]): The bounding box coordinates of the card.
            label (Any): The label associated with the detection.
        """
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.misses = 0

    def register_hit(self, bbox: Tuple[float, float, float, float], label: Any) -> None:
        """
        Update the track with a new detection.

        This method resets the miss count, increments the hit count, and updates the bounding box and label
        based on the new detection.

        Parameters:
            bbox  (Tuple[float, float, float, float]): The new bounding box.
            label (Any): The updated label for the card.
        """
        self.bbox = bbox
        self.label = label
        self.misses = 0
        self.hits += 1

    def register_miss(self) -> None:
        """
        Update the track with a miss.

        This method increments the miss count and resets the hit count.
        """
        self.misses += 1
        self.hits = 0


class CardTracker(ICardTracker):
    """
    CardTracker is an implementation of the ICardTracker interface.

    This implementation maintains active card tracks across frames by associating new detections to existing
    tracks via the Hungarian algorithm on IoU costs. It updates hits/misses, confirms tracks after consecutive
    hits, deletes stale tracks after misses, and invokes an optional callback on confirmation.
    """

    def __init__(
        self,
        confidence_threshold: float,
        iou_threshold: float,
        confirmation_frames: int = 5,
        removal_frames: int = 10,
        on_confirm_callback: Optional[Callable[[Track], None]] = None,
    ) -> None:
        """
        Initialize CardTracker with tracking parameters and an optional confirmation callback.

        A callback can be provided to react whenever a track is confirmed.

        Parameters:
            confidence_threshold (float): The minimum detection confidence to consider.
            iou_threshold (float): The minimum IoU for matching a detection to a track.
            confirmation_frames (int): The hits required to confirm a new track.
            removal_frames (int): The misses required before deleting a track.
            on_confirm_callback (Optional[Callable[[Track], None]]): Function called with the Track when
            confirmed.
        """
        self.tracks = {}
        self.next_track_id = 0
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.confirmation_frames = confirmation_frames
        self.removal_frames = removal_frames
        self.on_confirm_callback = on_confirm_callback

    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute the Intersection over Union (IoU) between two box sets.

        This method handles reshaping of inputs, computes pairwise intersection areas, and divides by the union
        areas to produce an IoU matrix.

        Parameters:
            boxes1 (np.ndarray): An array of bounding boxes (shape: [N, 4]).
            boxes2 (np.ndarray): An array of bounding boxes (shape: [M, 4]).

        Returns:
            np.ndarray: An IoU matrix of shape (N, M).
        """
        # Ensure boxes are 2-dimensional arrays (N, 4)
        if boxes1.ndim != 2:
            boxes1 = boxes1.reshape(-1, 4)
        if boxes2.ndim != 2:
            boxes2 = boxes2.reshape(-1, 4)

        # Handle edge case where there are no boxes in one of the arrays
        if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]))

        # Compute the coordinates for the intersection rectangle
        xA = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        yA = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        xB = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        yB = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        # Calculate area of intersection rectangle and handle negative values
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

        # Compute individual box areas for union calculation
        boxArea1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxArea2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute IoU using the formula: Intersection / (Area1 + Area2 - Intersection)
        iou = interArea / (boxArea1[:, None] + boxArea2[None, :] - interArea + 1e-6)

        return iou

    def _data_association(
        self, detection_boxes: List[Tuple[float, float, float, float]]
    ) -> Tuple[Dict[int, int], Set[int]]:
        """
        Perform data association using the Hungarian algorithm on IoU cost.

        This method converts IoU to a cost matrix (1 - IoU), solves the assignment problem, and then filters
        matches below the IoU threshold. Unmatched detections are returned separately.

        Parameters:
            detection_boxes (List[Tuple[float, float, float, float]]): A list of bounding boxes.

        Returns:
            Tuple[Dict[int, int], Set[int]]: A mapping of track IDs to indices of detection boxes and a set of
            indices representing unmatched detections.
        """
        # If there are no new detections, return empty assignments and an empty set for unmatched detections
        if len(detection_boxes) == 0:
            return {}, set()

        # Prepare arrays for existing track bounding boxes and new detection bounding boxes
        track_boxes = np.array([track.bbox for track in self.tracks.values()])
        det_boxes = np.array(detection_boxes).reshape(-1, 4)

        # Compute the IoU matrix between each track and detection
        iou_matrix = self._compute_iou(track_boxes, det_boxes)
        # Convert IoU to a cost matrix for minimization; lower cost means higher overlap
        cost_matrix = 1 - iou_matrix

        # Solve the assignment problem using the Hungarian algorithm (minimizes total cost)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = {}  # To store valid assignments
        unmatched_detections = set(
            range(len(detection_boxes))
        )  # Start with all detections as unmatched
        track_ids = list(
            self.tracks.keys()
        )  # Get current track IDs to index into assignments

        # Loop through each assignment provided by the algorithm
        for r, c in zip(row_ind, col_ind):
            # Only accept the assignment if the IoU meets or exceeds the threshold
            if iou_matrix[r, c] >= self.iou_threshold:
                track_id = track_ids[r]
                assignments[track_id] = c
                unmatched_detections.remove(c)

        return assignments, unmatched_detections

    def _update_tracks(
        self,
        assignments: Dict[int, int],
        unmatched_detections: Set[int],
        detection_boxes: List[Tuple[float, float, float, float]],
        detections: Dict[Tuple[float, float, float, float], Dict[str, Any]],
    ) -> None:
        """
        Update existing tracks based on matched and unmatched detections.

        This method registers hits for assigned tracks (confirming if hits reach the threshold), registers
        misses for others (deleting when too many), and spawns new Track objects for any unmatched detections.

        Parameters:
            assignments (Dict[int, int]): A mapping of track IDs to their corresponding detection index.
            unmatched_detections (Set[int]): A set of detection indices with no assignment.
            detection_boxes (List[Tuple[...]]): A list of detection boxes.
            detections (Dict[Tuple, Dict[str, Any]]): A mapping of bounding box coordinates to their detection
            information.
        """
        # Process tracks that have been assigned a detection
        for track_id, det_idx in assignments.items():
            track = self.tracks[track_id]
            detection_bbox = detection_boxes[det_idx]
            detection_info = detections.get(tuple(detection_bbox), {})
            detection_label = detection_info.get("label", None)

            track.register_hit(detection_bbox, detection_label)

            if (
                track.state == TrackState.TENTATIVE
                and track.hits >= self.confirmation_frames
            ):
                track.state = TrackState.CONFIRMED

                if self.on_confirm_callback:
                    self.on_confirm_callback(track)

        # Process tracks that did not receive a matching detection in the current frame
        for track_id in list(self.tracks.keys()):
            if track_id not in assignments:
                self.tracks[track_id].register_miss()

                if self.tracks[track_id].misses > self.removal_frames:
                    del self.tracks[track_id]

        # Create new tracks for any detection that was not matched to an existing track
        for det_idx in unmatched_detections:
            new_bbox = detection_boxes[det_idx]
            detection_info = detections.get(tuple(new_bbox), {})
            detection_label = detection_info.get("label", None)
            self.tracks[self.next_track_id] = Track(
                track_id=self.next_track_id, bbox=new_bbox, label=detection_label
            )
            self.next_track_id += 1

    def update(
        self, detections: Dict[Tuple, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Update tracked cards using new detections.

        This implementation converts detection bounding boxes into a list, performs data association and track
        updates, and returns the assembled mapping.

        Parameters:
            detections (Dict[Tuple, Dict[str, Any]]): A mapping of bounding box coordinates to their detection
            information.

        Returns:
            Dict[int, Dict[str, Any]]: A mapping of track IDs to their tracking information.

            - Key (int): A unique ID for each track.
            - Value (Dict[str, Any]): A dict of tracking information.
                - "bbox" (Tuple[float, float, float, float]): The current bounding box for this track.
                - "label" (Any): The label associated with the detection that created or updated this track.
                - "state" (int): The current track state.
        """
        detection_boxes = [list(bbox) for bbox in detections.keys()]
        assignments, unmatched_detections = self._data_association(detection_boxes)
        self._update_tracks(
            assignments, unmatched_detections, detection_boxes, detections
        )
        # Return the current state of all tracks
        return {
            tid: {"bbox": track.bbox, "label": track.label, "state": track.state}
            for tid, track in self.tracks.items()
        }
