from psrc.annotation.cv_annotator import CVAnnotator
from psrc.config.config_manager import ConfigManager
from psrc.core.analysis_engine import AnalysisEngine
from psrc.detection.card_detector import CardDetector
from psrc.detection.card_tracker import CardTracker
from psrc.detection.hand_tracker import HandTracker
from psrc.evaluation.card_deck import CardDeck
from psrc.evaluation.ev_calculator_wrapper import EVCalculatorWrapper
from psrc.ui.cv_display import CVDisplay
from psrc.video.cv_video_stream import CVVideoStream

def main():
  """
  Main entry point for the blackjack vision application.

  This function is used to load configuration settings to be passed in to the analysis engine. It initializes
  all required componenets (detector, tracker, deck, EV calculator, hand tracker, annotator, display) and passes
  them to the analysis engine.
  """
  # Load configuration from config.yaml
  settings = ConfigManager()
  
  # Determine the video source based on user preference (webcam or file)
  if settings.use_webcam:
    video_reader = CVVideoStream(source=settings.webcam_index)
  else:
    video_reader = CVVideoStream(source=settings.video_path)
  
  # Initialize the card detector with the YOLO model path
  card_detector = CardDetector(model_path=settings.yolo_path)
  
  # Initialize a deck with the specified size
  deck = CardDeck(deck_count=settings.deck_count)
  
  # Initialize the card tracker, which removes a card from the deck once it becomes locked
  card_tracker = CardTracker(
    confidence_threshold=settings.confidence_threshold,
    iou_threshold=settings.iou_threshold,
    confirmation_frames=settings.confirmation_frames,
    miss_frames=settings.removal_frames,
    on_confirm_callback=lambda track: deck.remove_card(track.label)
  )
  
  # Initialize the EV calculator with the jar and class paths
  ev_calculator = EVCalculatorWrapper(
    jar_path=settings.ev_jar_path,
    java_class=settings.ev_class_path
  )
  
  # Initialize the hand tracker 
  hand_tracker = HandTracker()
  
  # Initialize the annotator with the annotation settings
  annotator = CVAnnotator(
    confirmed_color=settings.confirmed_color,
    tentative_color=settings.tentative_color,
    font_scale=settings.font_scale,
    thickness=settings.thickness
  )
  
  # Initialize the display with the window name
  vision_display = CVDisplay(window_name=settings.window_name)
  
  # Load the analysis engine that ties the components together
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
  
  # Start the main analysis loop
  engine.run()

if __name__ == "__main__":
  main()
