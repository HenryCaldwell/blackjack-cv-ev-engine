from typing import Any, Dict, Tuple

from ultralytics import YOLO

from psrc.core.interfaces.i_card_detector import ICardDetector


class CardDetector(ICardDetector):
    """
    CardDetector is an implementation of the ICardDetector interface.

    This implementation wraps an Ultralytics YOLO model to run inference on individual frames and extract
    bounding boxes, class labels, and confidence scores.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize CardDetector with a YOLO model.

        Parameters:
            model_path (str): A filepath to pretrained YOLO weights.
        """
        self.model = YOLO(model_path)

    def detect(self, frame: Any) -> Dict[Tuple, Dict[str, Any]]:
        """
        Detect cards within a given frame.

        This implementation runs the YOLO model on the provided frame, extracts the bounding boxes, class
        indices, and confidence scores, converts them into Python-native lists, and returns the assembled
        mapping.

        Parameters:
            frame (Any): The frame in which to detect.

        Returns:
            Dict[Tuple, Dict[str, Any]]: A mapping of bounding box coordinates to their detection information.

            - Key (Tuple[float, float, float, float]): A tuple representing the bounding box.
                - (x1, y1) = top-left corner
                - (x2, y2) = bottom-right corner
            - Value (Dict[str, Any]): A dict of detection information.
                - "label" (Optional[int]): Class index assigned by YOLO for this box, or None.
                - "confidence" (float): Confidence score for this detection.
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

            if hasattr(last_results.boxes, "cls"):
                labels = last_results.boxes.cls.cpu().numpy().tolist()

        # Dictionary to store current detection information
        detections = {}

        for i in range(len(boxes)):
            # Convert each box to a tuple to use as a key
            detections[tuple(boxes[i])] = {
                "label": labels[i] if i < len(labels) else None,
                "confidence": confidences[i],
            }

        return detections
