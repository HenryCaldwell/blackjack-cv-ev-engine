from psrc.annotation.annotator import Annotator
from psrc.config.config_manager import ConfigManager
from psrc.core.analysis_engine import AnalysisEngine
from psrc.detection.card_detector import CardDetector
from psrc.detection.card_tracker import CardTracker
from psrc.detection.hand_tracker import HandTracker
from psrc.evaluation.card_deck import CardDeck
from psrc.evaluation.ev_calculator_wrapper import EVCalculatorWrapper
from psrc.ui.cv_display import CVDisplay
from psrc.video.video_stream import VideoStreamReader

def main():
  settings = ConfigManager()
  
  if settings.use_webcam:
    video_reader = VideoStreamReader(settings.webcam_index)
  else:
    video_reader = VideoStreamReader(settings.video_path)
  
  card_detector = CardDetector(settings.yolo_path)
  
  card_tracker = CardTracker(
    confirmation_frames=settings.confirmation_frames,
    disappear_frames=settings.disappear_frames,
    confidence_threshold=settings.confidence_threshold,
    overlap_threshold=settings.inference_overlap_threshold,
    on_lock_callback=lambda card_label: deck.remove_card(card_label)
  )
  
  deck = CardDeck(settings.deck_size)
  
  ev_calculator = EVCalculatorWrapper(
    jar_path=settings.ev_jar_path,
    java_class=settings.ev_class_path
  )
  
  hand_tracker = HandTracker()
  
  annotator = Annotator(
    player_color=settings.player_color,
    dealer_color=settings.dealer_color,
    default_color=settings.default_color,
    font_scale=settings.font_scale,
    thickness=settings.thickness
  )
  
  vision_display = CVDisplay(window_name=settings.window_name)
  
  engine = AnalysisEngine(
    video_reader=video_reader,
    card_detector=card_detector,
    card_tracker=card_tracker,
    deck=deck,
    ev_calculator=ev_calculator,
    hand_tracker=hand_tracker,
    annotator=annotator,
    vision_display=vision_display,
    inference_interval=settings.inference_interval,
    inference_frame_size=tuple(settings.inference_frame_size),
    display_frame_size=tuple(settings.display_frame_size)
  )
  
  engine.run()

if __name__ == "__main__":
  main()
