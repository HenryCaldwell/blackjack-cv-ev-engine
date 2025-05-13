from psrc.annotation.cv_annotator import CVAnnotator
from psrc.config.config_manager import ConfigManager
from psrc.core.analysis_engine import AnalysisEngine
from psrc.detection.card_detector import CardDetector
from psrc.detection.card_tracker import CardTracker
from psrc.detection.hand_tracker import HandTracker
from psrc.evaluation.card_deck import CardDeck
from psrc.evaluation.ev_calculator_wrapper import EVCalculatorWrapper
from psrc.evaluation.hand_evaluator import HandEvaluator
from psrc.ui.cv_display import CVDisplay
from psrc.video.cv_video_stream import CVVideoStream


def main():
  """
  Main entry point for the blackjack vision application.

  This function is used to load configuration settings to be passed in to the analysis engine. It initializes
  all required componenets and passes them to the analysis engine.
  """
  settings = ConfigManager()

  # Core Components
  source = settings.webcam_index if settings.use_webcam else settings.video_path
  video_reader = CVVideoStream(source=source)

  deck = CardDeck(deck_count=settings.deck_count)

  card_detector = CardDetector(model_path=settings.yolo_path)

  card_tracker = CardTracker(
    confidence_threshold=settings.confidence_threshold,
    iou_threshold=settings.iou_threshold,
    confirmation_frames=settings.confirmation_frames,
    miss_frames=settings.removal_frames,
    on_confirm_callback=lambda track: deck.remove_card(track.label),
  )

  hand_tracker = HandTracker()
  
  ev_calculator = EVCalculatorWrapper(
    jar_path=settings.ev_jar_path, java_class=settings.ev_class_path
  )

  hand_evaluator = HandEvaluator(
    deck=deck,
    ev_calculator=ev_calculator
  )

  annotator = CVAnnotator(
    confirmed_color=settings.confirmed_color,
    tentative_color=settings.tentative_color,
    font_scale=settings.font_scale,
    thickness=settings.thickness,
  )

  vision_display = CVDisplay(window_name=settings.window_name)

  # Core Engine
  engine = AnalysisEngine(
    video_reader=video_reader,
    card_detector=card_detector,
    card_tracker=card_tracker,
    hand_evaluator=hand_evaluator,
    hand_tracker=hand_tracker,
    annotator=annotator,
    vision_display=vision_display,
    inference_interval=settings.inference_interval,
    inference_frame_size=tuple(settings.inference_frame_size),
    display_frame_size=tuple(settings.display_frame_size),
  )

  engine.run()

if __name__ == "__main__":
  main()
