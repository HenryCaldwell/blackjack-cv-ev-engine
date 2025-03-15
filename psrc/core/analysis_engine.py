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

    self.last_update = 0.0

  def evaluate_hands(self, hands_info: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    dealer_cards = []
    
    if "Dealer" in hands_info:
      dealer_cards = [int(card) for card in hands_info["Dealer"].get("cards", [])]

    for hand_id, info in hands_info.items():
      if hand_id == "Dealer":
        continue
      
      player_cards = [int(card) for card in info.get("cards", [])]
      evs = {}
      actions = ["stand", "hit", "double", "split"]
      for action in actions:
        try:
          ev = self.ev_calculator.calculate_ev(
            action, self.deck.cards, player_cards, dealer_cards
          )
          evs[action] = ev
        except Exception as e:
          logger.error("Error evaluating action '%s' for %s (%s): %s",
                            action, hand_id, player_cards, e)
      if evs:
        best_action = max(evs, key=evs.get)
        results[hand_id] = {"evs": evs, "best_action": best_action}
      else:
        results[hand_id] = {"evs": None, "best_action": None}

    return results


  def run(self) -> None:
    logger.info("Starting processing loop")

    while True:
        frame = self.video_reader.read_frame()

        if frame is None:
          logger.info("No frame received, exiting processing loop")
          break

        inference_frame = cv2.resize(frame, self.inference_frame_size)
        current_time = time.time()

        if current_time - self.last_update >= self.inference_interval:
          raw_detections = self.card_detector.detect(inference_frame)
          tracked_detections = self.card_tracker.update(raw_detections)
          hands_info = self.hand_tracker.update(tracked_detections)
          
          for hand, info in hands_info.items():
            logger.info("%s: Cards: %s | Score: %s", hand, info["cards"], info["score"])
          
          ev_results = self.evaluate_hands(hands_info)
          
          for hand, data in ev_results.items():
            formatted_evs = {action: f"{(value * 100):.2f}%" for action, value in data["evs"].items()}
            logger.info("%s: EVs: %s | Best: %s", hand, formatted_evs, data["best_action"].upper())

          annotated_frame = self.annotator.annotate(inference_frame.copy(), raw_detections.keys(), hands_info)
          self.last_update = current_time

        display_frame = cv2.resize(annotated_frame, self.display_frame_size)
        self.vision_display.update_and_show(display_frame)

        if not self.vision_display.handle_input():
          logger.info("Quit signal received; exiting loop")
          break

    self.video_reader.release()
    logger.info("Video input stream released")
    self.ev_calculator.release()
    logger.info("JVM released")
    self.vision_display.release()
    logger.info("Video output stream released")
    logger.info("AnalysisEngine terminated")