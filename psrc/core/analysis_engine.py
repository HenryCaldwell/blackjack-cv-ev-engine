import cv2
import time

from typing import Tuple
from psrc.core.interfaces.i_annotator import IAnnotator
from psrc.core.interfaces.i_card_detector import ICardDetector
from psrc.core.interfaces.i_card_tracker import ICardTracker
from psrc.core.interfaces.i_display import IDisplay
from psrc.core.interfaces.i_ev_calculator import IEVCalculator
from psrc.core.interfaces.i_hand_evaluator import IHandEvaluator
from psrc.core.interfaces.i_hand_tracker import IHandTracker
from psrc.core.interfaces.i_video_stream import IVideoStreamReader

from psrc.debugging.logger import setup_logger
logger = setup_logger(__name__)

class AnalysisEngine:
  """
  AnalysisEngine orchestrates the video processing pipeline for card detection, tracking, hand grouping, and
  expected value calculations in a blackjack context.

  It reads frames from a video source, detects and tracks cards, computes hand scores, evaluates the expected
  value for various player actions, and displays the annotated frames.
  """
  
  def __init__(self,
                video_reader: IVideoStreamReader,
                card_detector: ICardDetector,
                card_tracker: ICardTracker,
                hand_tracker: IHandTracker,
                hand_evaluator: IHandEvaluator,
                annotator: IAnnotator,
                vision_display: IDisplay,
                inference_interval: float = 0.25,
                inference_frame_size: Tuple[int, int] = (1280, 720),
                display_frame_size: Tuple[int, int] = (1280, 720)) -> None:
    """
    Initialize the AnalysisEngine with all the components required for the video processing workflow.

    Parameters:
      video_reader (IVideoStreamReader): The video reader instance for retrieving frames.
      card_detector (ICardDetector): The card detector for identifying cards in frames.
      card_tracker (ICardTracker): The card tracker for maintaining detection continuity.
      hand_tracker (IHandTracker): The hand tracker for grouping cards and computing hand scores.
      hand_evaluator (IHandEvaluator): Evaluates hands and selects optimal actions.
      annotator (IAnnotator): The annotator for drawing bounding boxes and hand details on frames.
      vision_display (IDisplay): The display interface for showing annotated frames and handling user input.

      inference_interval (float): Minimum time (in seconds) between inference steps. Default is 0.25s.
      inference_frame_size (Tuple[int, int]): The resolution for inference processing. Default is (1920, 1080).
      display_frame_size (Tuple[int, int]): The resolution for display output. Default is (1280, 720).
    """
    self.video_reader = video_reader
    self.card_detector = card_detector
    self.card_tracker = card_tracker
    self.hand_tracker = hand_tracker
    self.hand_evaluator = hand_evaluator
    self.annotator = annotator
    self.vision_display = vision_display
    self.inference_interval = inference_interval
    self.inference_frame_size = inference_frame_size
    self.display_frame_size = display_frame_size

    # Track the last time we performed inference
    self.last_update = 0.0

  def run(self) -> None:
    """
    Execute the main processing loop for video analysis.

    This method continuously reads frames from the video reader, resizes them for inference, and runs card
    detection and tracking at a specified interval. It then groups cards into hands, logs their scores,
    calculates EV for different actions, and annotates the frame. Finally, it displays the annotated frame and
    checks for user input to quit the loop.
    """
    logger.info("Starting processing loop")

    while True:
        frame = self.video_reader.read_frame()

        # If no frame is available, exit the loop
        if frame is None:
          logger.info("No frame received, exiting processing loop")
          break

        # Resize frame for inference
        inference_frame = cv2.resize(frame, self.inference_frame_size)
        current_time = time.time()

        # Perform detection and tracking if enough time has passed
        if current_time - self.last_update >= self.inference_interval:
          raw_detections = self.card_detector.detect(inference_frame)
          tracked_detections = self.card_tracker.update(raw_detections)
          hands_info = self.hand_tracker.update(tracked_detections)
          eval_results = self.hand_evaluator.evaluate_hands(hands_info)

          logger.info("Evaluation results: %s", eval_results)

          annotated_frame = self.annotator.annotate(inference_frame.copy(), raw_detections.keys(), tracked_detections)
          self.last_update = current_time

        display_frame = cv2.resize(annotated_frame, self.display_frame_size)
        self.vision_display.update(display_frame)

        # Check for user input to exit
        if not self.vision_display.handle_input():
          logger.info("Quit signal received; exiting loop")
          break

     # Release resources
    self.video_reader.release()
    logger.info("Video input stream released")
    self.vision_display.release()
    logger.info("Video output stream released")
    logger.info("AnalysisEngine terminated")