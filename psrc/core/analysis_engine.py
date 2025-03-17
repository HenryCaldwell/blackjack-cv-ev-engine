import cv2
import time

from typing import Any, Dict, Tuple
from psrc.core.interfaces.i_annotator import IAnnotator
from psrc.core.interfaces.i_card_deck import ICardDeck
from psrc.core.interfaces.i_card_detector import ICardDetector
from psrc.core.interfaces.i_card_tracker import ICardTracker
from psrc.core.interfaces.i_display import IDisplay
from psrc.core.interfaces.i_ev_calculator import IEVCalculator
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
                deck: ICardDeck,
                ev_calculator: IEVCalculator,
                hand_tracker: IHandTracker,
                annotator: IAnnotator,
                vision_display: IDisplay,
                inference_interval: float = 0.25,
                inference_frame_size: Tuple[int, int] = (1920, 1080),
                display_frame_size: Tuple[int, int] = (1280, 720)) -> None:
    """
    Initialize the AnalysisEngine with all the components required for the video processing workflow.

    Parameters:
      video_reader (IVideoStreamReader): The video reader instance for retrieving frames.
      card_detector (ICardDetector): The card detector for identifying cards in frames.
      card_tracker (ICardTracker): The card tracker for maintaining detection continuity.
      deck (ICardDeck): The card deck manager for tracking deck composition.
      ev_calculator (IEVCalculator): The expected value calculator for blackjack actions.
      hand_tracker (IHandTracker): The hand tracker for grouping cards and computing hand scores.
      annotator (IAnnotator): The annotator for drawing bounding boxes and hand details on frames.
      vision_display (IDisplay): The display interface for showing annotated frames and handling user input.

      inference_interval (float): Minimum time (in seconds) between inference steps. Default is 0.25s.
      inference_frame_size (Tuple[int, int]): The resolution for inference processing. Default is (1920, 1080).
      display_frame_size (Tuple[int, int]): The resolution for display output. Default is (1280, 720).
    """
    self.video_reader = video_reader
    self.card_detector = card_detector
    self.card_tracker = card_tracker
    self.deck = deck
    self.ev_calculator = ev_calculator
    self.hand_tracker = hand_tracker
    self.annotator = annotator
    self.vision_display = vision_display
    self.inference_interval = inference_interval
    self.inference_frame_size = inference_frame_size
    self.display_frame_size = display_frame_size

    # Track the last time we performed inference
    self.last_update = 0.0

  def evaluate_hands(self, hands_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the expected value (EV) of various blackjack actions for the detected player hands.

    This method retrieves the dealer's cards, then for each player hand, attempts to calculate the EV of
    "stand", "hit", "double", and "split" actions. It logs errors if any action calculation fails. The best
    action is determined by the highest EV.

    Parameters:
      hands_info (Dict[str, Any]): A dictionary mapping hand identifiers to their details (e.g., cards, score,
      boxes).

    Returns:
      Dict[str, Any]: A dictionary of evaluation results, mapping hand identifiers to a structure containing EVs
      for each action and the best action.
    """
    results = {}
    dealer_cards = []
    
    # Extract dealer cards if available
    if "Dealer" in hands_info:
      dealer_cards = [int(card) for card in hands_info["Dealer"].get("cards", [])]

    # Iterate over each hand, skipping the dealer
    for hand_id, info in hands_info.items():
      if hand_id == "Dealer":
        continue
      
      player_cards = [int(card) for card in info.get("cards", [])]
      evs = {}
      actions = ["stand", "hit", "double", "split"]

      # Compute EV for each action
      for action in actions:
        try:
          ev = self.ev_calculator.calculate_ev(
            action, self.deck.cards, player_cards, dealer_cards
          )
          evs[action] = ev
        except Exception as e:
          logger.error("Error evaluating action '%s' for %s (%s): %s",
                            action, hand_id, player_cards, e)
      
      # Determine the best action if EVs are available
      if evs:
        best_action = max(evs, key=evs.get)
        results[hand_id] = {"evs": evs, "best_action": best_action}
      else:
        results[hand_id] = {"evs": None, "best_action": None}

    return results


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
          
          for hand, info in hands_info.items():
            logger.info("%s: Cards: %s | Score: %s", hand, info["cards"], info["score"])
          
          ev_results = self.evaluate_hands(hands_info)
          
          # Evaluate hands to determine best actions
          for hand, data in ev_results.items():
            formatted_evs = {action: f"{(value * 100):.2f}%" for action, value in data["evs"].items()}
            logger.info("%s: EVs: %s | Best: %s", hand, formatted_evs, data["best_action"].upper())

          annotated_frame = self.annotator.annotate(inference_frame.copy(), raw_detections.keys(), hands_info)
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
    self.ev_calculator.release()
    logger.info("JVM released")
    self.vision_display.release()
    logger.info("Video output stream released")
    logger.info("AnalysisEngine terminated")