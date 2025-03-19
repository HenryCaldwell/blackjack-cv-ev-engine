import numpy as np

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
    track_id (int): Unique identifier for the track.
    bbox (Tuple[float, float, float, float]): Bounding box coordinates of the card.
    state (int): Current state of the track (TENTATIVE, CONFIRMED, or DELETED).
    hits (int): Number of consecutive frames where the card was detected.
    misses (int): Number of consecutive frames where the card was not detected.
    label (Any): Label or category associated with the detection.
  """

  def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], label: Any) -> None:
    """
    Initialize a new Track instance.

    Parameters:
      track_id (int): Unique identifier for the track.
      bbox (Tuple[float, float, float, float]): Initial bounding box coordinates.
      label (Any): Label detected for the card.
    """
    self.track_id = track_id
    self.bbox = bbox
    self.label = label
    self.state = TrackState.TENTATIVE
    self.hits = 1
    self.misses = 0

  def register_hit(self, new_bbox: Tuple[float, float, float, float], new_label: Any) -> None:
    """
    Update the track with a new detection.

    Resets the miss count, increments the hit count, and updates the bounding box and label.

    Parameters:
      new_bbox (Tuple[float, float, float, float]): New bounding box coordinates.
      new_label (Any): Updated label for the card.
    """
    self.bbox = new_bbox
    self.label = new_label
    self.misses = 0
    self.hits += 1

  def register_miss(self) -> None:
    """
    Update the track with a miss when no detection is associated.

    Increments the miss count and resets the hit count.
    """
    self.misses += 1
    self.hits = 0

class CardTracker(ICardTracker):
  """
  CardTracker implements the ICardTracker interface for tracking card detections across video frames.

  This class uses data association via the Hungarian algorithm to match new detections with existing tracks
  based on Intersection over Union (IoU). Tracks are updated based on hits and misses, and are confirmed after a
  specified number of consecutive hits.
  """

  def __init__(self,
                confidence_threshold: float,
                iou_threshold: float,
                confirmation_frames: int = 5, 
                miss_frames: int = 10,
                on_confirm_callback: Optional[Callable[[Track], None]] = None) -> None:
    """
    Initialize the CardTracker with tracking parameters and an optional callback.

    Parameters:
      confidence_threshold (float): Minimum confidence required to consider a detection valid.
      iou_threshold (float): Minimum IoU required for matching a detection with an existing track.
      confirmation_frames (int): Minimum number of consecutive hits required to confirm a track.
      miss_frames (int): Maximum allowed consecutive missed detections before deleting a track.
      on_confirm_callback (Optional[Callable[[Track], None]]): Function to call when a track is confirmed.
    """
    self.tracks = {}  # Dictionary to store active tracks with track_id as keys
    self.next_track_id = 0
    self.confidence_threshold = confidence_threshold
    self.iou_threshold = iou_threshold
    self.confirmation_frames = confirmation_frames
    self.miss_frames = miss_frames
    self.on_confirm_callback = on_confirm_callback

  def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Parameters:
      boxes1 (np.ndarray): Array of bounding boxes (shape: [N, 4]).
      boxes2 (np.ndarray): Array of bounding boxes (shape: [M, 4]).

    Returns:
      np.ndarray: IoU matrix of shape (N, M) where each element represents the IoU between boxes.
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

  def _data_association(self, detection_boxes: List[Tuple[float, float, float, float]]) -> Tuple[Dict[int, int], Set[int]]:
    """
    Perform data association between existing tracks and new detections using the Hungarian algorithm.

    Computes the IoU between track bounding boxes and new detection boxes. Detections are assigned to tracks if
    the IoU is above the given threshold.

    Parameters:
      detection_boxes (List[Tuple[float, float, float, float]]): List of bounding boxes from new detections.

    Returns:
      Tuple[Dict[int, int], Set[int]]:
        - A dictionary mapping track IDs to indices of detection boxes.
        - A set of indices representing unmatched detections.
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
    unmatched_detections = set(range(len(detection_boxes)))  # Start with all detections as unmatched
    track_ids = list(self.tracks.keys())  # Get current track IDs to index into assignments
    
    # Loop through each assignment provided by the algorithm
    for r, c in zip(row_ind, col_ind):
      # Only accept the assignment if the IoU meets or exceeds the threshold
      if iou_matrix[r, c] >= self.iou_threshold:
        track_id = track_ids[r]
        assignments[track_id] = c
        unmatched_detections.remove(c)
    
    return assignments, unmatched_detections

  def _update_tracks(self,
                     assignments: Dict[int, int],
                     unmatched_detections: Set[int],
                     detection_boxes: List[Tuple[float, float, float, float]],
                     detections: Dict[Tuple[float, float, float, float], Dict[str, Any]]) -> None:
    """
    Update existing tracks based on the assignment of detections.

    For tracks with an assigned detection, update the bounding box and label, and increment hit counts. If a
    tentative track reaches the confirmation threshold, mark it as confirmed and trigger the lock callback.
    Tracks that are not assigned a detection register a miss will be removed if they exceed the allowed misses.
    Unmatched detections start new tracks.

    Parameters:
      assignments (Dict[int, int]): Mapping of track IDs to detection indices.
      unmatched_detections (Set[int]): Set of detection indices that were not assigned to any track.
      detection_boxes (List[Tuple[float, float, float, float]]): List of detection bounding boxes.
      detections (Dict[Tuple[float, float, float, float], Dict[str, Any]]): Dictionary of detection details
      keyed by bounding box.
    """
    # Process tracks that have been assigned a detection
    for track_id, det_idx in assignments.items():
      track = self.tracks[track_id]
      detection_bbox = detection_boxes[det_idx]
      detection_info = detections.get(tuple(detection_bbox), {})
      detection_label = detection_info.get("label", None)
      
      track.register_hit(detection_bbox, detection_label)

      if track.state == TrackState.TENTATIVE and track.hits >= self.confirmation_frames:
        track.state = TrackState.CONFIRMED

        if self.on_confirm_callback:
          self.on_confirm_callback(track)

    # Process tracks that did not receive a matching detection in the current frame
    for track_id in list(self.tracks.keys()):
      if track_id not in assignments:
        self.tracks[track_id].register_miss()

        if self.tracks[track_id].misses > self.miss_frames:
          del self.tracks[track_id]

    # Create new tracks for any detection that was not matched to an existing track
    for det_idx in unmatched_detections:
      new_bbox = detection_boxes[det_idx]
      detection_info = detections.get(tuple(new_bbox), {})
      detection_label = detection_info.get("label", None)
      self.tracks[self.next_track_id] = Track(
        track_id=self.next_track_id,
        bbox=new_bbox,
        label=detection_label
      )
      self.next_track_id += 1

  def update(self, detections: Dict[Tuple[float, float, float, float], Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Update tracked card information using new detections.

    This method converts detection bounding boxes into a list, performs data association with existing tracks,
    and updates each track based on matches and misses.

    Parameters:
      detections (Dict[Tuple[float, float, float, float], Dict[str, Any]]): Dictionary mapping detection
      bounding boxes to detection details.

    Returns:
      Dict[int, Dict[str, Any]]: Dictionary mapping track IDs to a dictionary containing the bounding box, label,
      and current state.
    """
    detection_boxes = [list(bbox) for bbox in detections.keys()]
    assignments, unmatched_detections = self._data_association(detection_boxes)
    self._update_tracks(assignments, unmatched_detections, detection_boxes, detections)
    # Return the current state of all tracks
    return {tid: {"bbox": track.bbox, "label": track.label, "state": track.state} 
            for tid, track in self.tracks.items()}