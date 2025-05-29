import os
import threading
import logging

from psrc.core.analysis_engine import AnalysisEngine
from psrc.evaluation.card_deck import CardDeck
from psrc.detection.card_detector import CardDetector
from psrc.detection.card_tracker import CardTracker
from psrc.config.config_manager import ConfigManager
from psrc.annotation.mpl_annotator import MPLAnnotator
from psrc.input.cv_video_stream import CVVideoStream
from psrc.evaluation.ev_calculator_wrapper import EVCalculatorWrapper
from psrc.evaluation.hand_evaluator import HandEvaluator
from psrc.detection.hand_tracker import HandTracker
from psrc.ui.hybrid_display import HybridDisplay

os.environ["YOLO_VERBOSE"] = "0"
logging.getLogger("psrc.core.analysis_engine").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def main() -> None:
    """
    Main entry point for the blackjack computer vision evaluation application.

    This function is used to load configuration settings to be passed in to the analysis engine. It initializes
    all required componenets and passes them to the analysis engine.
    """
    settings = ConfigManager()

    # Core Components
    source = settings.webcam_index if settings.use_webcam else settings.video_path
    video_reader = CVVideoStream(source=source)

    card_detector = CardDetector(model_path=settings.yolo_path)

    deck = CardDeck(settings.deck_count)

    tracker = CardTracker(
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        confirmation_frames=settings.confirmation_frames,
        miss_frames=settings.removal_frames,
        on_confirm_callback=lambda track: deck.remove_card(track.label),
    )

    hand_tracker = HandTracker()

    ev_calculator = EVCalculatorWrapper(
        jar_path=settings.ev_jar_path,
        java_class=settings.ev_class_path,
    )

    hand_evaluator = HandEvaluator(deck=deck, ev_calculator=ev_calculator)

    annotator = MPLAnnotator()

    display = HybridDisplay(
        window_name=settings.window_name, window_frame_size=settings.window_frame_size
    )

    # Core Engine
    engine = AnalysisEngine(
        video_reader=video_reader,
        card_detector=card_detector,
        card_tracker=tracker,
        hand_tracker=hand_tracker,
        deck=deck,
        hand_evaluator=hand_evaluator,
        annotator=annotator,
        display=display,
        inference_interval=settings.inference_interval,
        inference_frame_size=tuple(settings.inference_frame_size),
        annotation_frame_size=tuple(settings.annotation_frame_size),
    )

    # Engine Thread
    engine_thread = threading.Thread(target=engine.start)

    try:
        # Start Engine and Display
        engine_thread.start()
        display.start()
    finally:
        # Tear Down Resources
        engine_thread.join()
        video_reader.release()
        display.release()
        ev_calculator.release()


if __name__ == "__main__":
    main()
