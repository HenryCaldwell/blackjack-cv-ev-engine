import numpy as np

from typing import Any, Dict
from ultralytics import YOLO

from psrc.core.interfaces.i_card_detector import ICardDetector
from psrc.detection.detection_utils import compute_overlap

class CardDetector(ICardDetector):
  """
  CardDetector implements the ICardDetector interface for detecting cards in video frames.
  
  It leverages a YOLO model to perform object detection and applies non-maximal suppression (NMS) to eliminate
  overlapping detections based on a specified overlap threshold.
  """
  
  def __init__(self, model_path: str, overlap_threshold: float = 0.9) -> None:
    """
    Initialize the CardDetector with a YOLO model and an overlap threshold.
    
    Parameters:
      model_path (str): Path to the YOLO model file.
      overlap_threshold (float): Overlap threshold for non-maximal suppression.
    """
    self.model = YOLO(model_path)
    self.overlap_threshold = overlap_threshold

  def detect(self, frame: Any) -> Dict[tuple, Dict[str, Any]]:
    """
    Detect cards in the provided video frame.
    
    The method runs the YOLO model on the frame, extracts bounding boxes, confidence scores, and labels (if
    available), and then applies non-maximal suppression to filter redundant detections.
    
    Parameters:
      frame (Any): The image frame in which to detect cards.
    
    Returns:
      Dict[tuple, Dict[str, Any]]: A dictionary mapping bounding box coordinates (as tuples) to their detection
      information (e.g., label, confidence).
    """
    # Run the YOLO model on the frame
    results = self.model(frame, show=False)
    # Use the first result which contains the detection data
    last_results = results[0]
    boxes, raw_labels, confidences = [], [], []
    
    # Check if detection results and bounding boxes are available
    if last_results is not None and last_results.boxes is not None:
      boxes = last_results.boxes.xyxy.cpu().numpy().tolist()
      confidences = last_results.boxes.conf.cpu().numpy().tolist()
      
      if hasattr(last_results.boxes, 'cls'):
        raw_labels = last_results.boxes.cls.cpu().numpy().tolist()
    
    # Apply non-maximal suppression to remove overlapping detections
    detections = self.apply_nms(boxes, raw_labels, confidences)
    return detections

  def apply_nms(self, boxes: list, labels: list, confidences: list) -> Dict[tuple, Dict[str, Any]]:
    """
    Apply Non-Maximal Suppression (NMS) to remove redundant overlapping detections.
    
    For each detection, the method compares the overlap with all others and retains only those with the highest
    confidence scores, filtering out others that exceed the overlap threshold.
    
    Parameters:
      boxes (list): List of bounding box coordinates.
      labels (list): List of corresponding labels.
      confidences (list): List of detection confidence scores.
    
    Returns:
      Dict[tuple, Dict[str, Any]]: Dictionary of filtered detections mapping bounding box coordinates (as tuples)
      to their detection information (e.g., label, confidence).
    """
    if not boxes:
      return {}
    
    boxes_np = np.array(boxes)
    confidences_np = np.array(confidences)
    # Sort box indices by descending confidence scores
    indices = sorted(range(len(confidences_np)), key=lambda i: confidences_np[i], reverse=True)
    keep = []
    
    # Process detections, keeping the most confident ones and suppressing overlapping boxes
    while indices:
      i = indices.pop(0)
      keep.append(i)
      # Filter out indices where the overlap with the current box exceeds the threshold
      indices = [j for j in indices if compute_overlap(boxes_np[i], boxes_np[j]) < self.overlap_threshold]
    
    detections = {}
    for i in keep:
      detections[tuple(boxes_np[i].tolist())] = {
        "label": labels[i],
        "confidence": confidences_np[i]
      }

    return detections