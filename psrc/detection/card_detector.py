import numpy as np

from typing import Any, Dict
from ultralytics import YOLO

from psrc.core.interfaces.i_card_detector import ICardDetector
from psrc.detection.detection_utils import compute_overlap

class CardDetector(ICardDetector):
  def __init__(self, model_path: str, overlap_threshold: float = 0.9) -> None:
    self.model = YOLO(model_path)
    self.overlap_threshold = overlap_threshold

  def detect(self, frame: Any) -> Dict[tuple, Dict[str, Any]]:
    results = self.model(frame, show=False)
    last_results = results[0]
    boxes, raw_labels, confidences = [], [], []
    
    if last_results is not None and last_results.boxes is not None:
      boxes = last_results.boxes.xyxy.cpu().numpy().tolist()
      confidences = last_results.boxes.conf.cpu().numpy().tolist()
      
      if hasattr(last_results.boxes, 'cls'):
        raw_labels = last_results.boxes.cls.cpu().numpy().tolist()
    
    detections = self.apply_nms(boxes, raw_labels, confidences)
    return detections

  def apply_nms(self, boxes: list, labels: list, confidences: list) -> Dict[tuple, Dict[str, Any]]:
    if not boxes:
      return {}
    
    boxes_np = np.array(boxes)
    confidences_np = np.array(confidences)
    indices = sorted(range(len(confidences_np)), key=lambda i: confidences_np[i], reverse=True)
    keep = []
    
    while indices:
      i = indices.pop(0)
      keep.append(i)
      indices = [j for j in indices if compute_overlap(boxes_np[i], boxes_np[j]) < self.overlap_threshold]
    
    detections = {}
    for i in keep:
      detections[tuple(boxes_np[i].tolist())] = {
        "label": labels[i],
        "confidence": confidences_np[i]
      }

    return detections