import numpy as np

from typing import Any, Dict
from ultralytics import YOLO

from psrc.core.interfaces.i_card_detector import ICardDetector
# from psrc.detection.detection_utils import compute_overlap

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
    
    The method runs the YOLO model on the frame, extracting bounding boxes, confidence scores, and labels.
    
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
    boxes, labels, confidences = [], [], []
    
    # Check if detection results and bounding boxes are available
    if last_results is not None and last_results.boxes is not None:
      boxes = last_results.boxes.xyxy.cpu().numpy().tolist()
      confidences = last_results.boxes.conf.cpu().numpy().tolist()
      
      if hasattr(last_results.boxes, 'cls'):
        labels = last_results.boxes.cls.cpu().numpy().tolist()
    
    # Dictionary to store current detection information
    detections = {}

    for i in range(len(boxes)):
      # Convert each box to a tuple to use as a key
      detections[tuple(boxes[i])] = {
        "label": labels[i] if i < len(labels) else None,
        "confidence": confidences[i]
      }

    return detections