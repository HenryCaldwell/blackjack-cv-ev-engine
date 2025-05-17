import cv2
import threading
import time
import queue
from typing import Any, Optional, Tuple

from psrc.core.interfaces.i_annotator import IAnnotator
from psrc.core.interfaces.i_card_detector import ICardDetector
from psrc.core.interfaces.i_card_tracker import ICardTracker
from psrc.core.interfaces.i_hand_tracker import IHandTracker
from psrc.core.interfaces.i_hand_evaluator import IHandEvaluator
from psrc.core.interfaces.i_display import IDisplay
from psrc.core.interfaces.i_video_stream import IVideoStreamReader

from psrc.debugging.logger import setup_logger

logger = setup_logger(__name__)


class AnalysisEngine:
    """
    AnalysisEngine orchestrates the video processing pipeline for card detection, tracking, hand grouping, and
    expected value calculations in a blackjack context.

    It runs three concurrent threads:
      1. Capture thread reads frames at source FPS into a single-entry queue.
      2. Analysis thread polls that queue at a configurable interval to run detection, tracking, grouping, and EV
      evaluation, then enqueues processed frames and metadata.
      3. Display thread dequeues processed items, overlays annotations, renders via display, and handles quit.
    """

    def __init__(
        self,
        video_reader: IVideoStreamReader,
        card_detector: ICardDetector,
        card_tracker: ICardTracker,
        hand_tracker: IHandTracker,
        hand_evaluator: IHandEvaluator,
        annotator: IAnnotator,
        display: IDisplay,
        inference_interval: float = 0.25,
        inference_frame_size: Tuple[int, int] = (1280, 720),
        display_frame_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        """
         Initialize the AnalysisEngine with all the components required for the video processing workflow.

        Parameters:
          video_reader (IVideoStreamReader): The video reader instance for retrieving frames.
          card_detector (ICardDetector): The card detector for identifying cards in frames
          card_tracker (ICardTracker): The card tracker for maintaining detection continuity
          hand_tracker (IHandTracker): The hand tracker for grouping cards and computing hand scores.
          hand_evaluator (IHandEvaluator): Evaluates hands and selects optimal actions.
          annotator (IAnnotator): The annotator for drawing bounding boxes and hand details on frames.
          display (IDisplay): The display interface for showing annotated frames and handling user input.

          inference_interval (float): Minimum time (in seconds) between inference steps.
          inference_frame_size (Tuple[int, int]): The resolution for inference processing.
          display_frame_size (Tuple[int, int]): The resolution for display output.
        """
        # Core components
        self.video_reader = video_reader
        self.card_detector = card_detector
        self.card_tracker = card_tracker
        self.hand_tracker = hand_tracker
        self.hand_evaluator = hand_evaluator
        self.annotator = annotator
        self.display = display

        # Timing & sizes
        self.inference_interval = inference_interval
        self.inference_frame_size = inference_frame_size
        self.display_frame_size = display_frame_size

        # Queues for thread data
        self.raw_queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=1)
        self.proc_queue: queue.Queue[Optional[Tuple]] = queue.Queue(maxsize=1)

        # Control
        self.running = False
        self.last_inference = 0.0

    def _capture_loop(self) -> None:
        """
        Continuously read frames from the video source at its native FPS and enqueue the latest frame.
        """
        fps = self.video_reader.get_fps()
        frame_period = 1.0 / fps

        while self.running:
            start = time.time()
            frame = self.video_reader.read_frame()
            if frame is None:
                self._enqueue_safe(self.raw_queue, None)
                break

            self._enqueue_safe(self.raw_queue, frame)
            time.sleep(max(0.0, frame_period - (time.time() - start)))

    def _analysis_loop(self) -> None:
        """
        At fixed intervals, pull the latest raw frame, run detection, tracking, hand grouping, and EV evaluation,
        then enqueue the processed frame and metadata for display.
        """
        while self.running:
            try:
                frame = self.raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                self._enqueue_safe(self.proc_queue, None)
                self.running = False
                break

            now = time.time()
            if now - self.last_inference < self.inference_interval:
                continue

            # Preprocess
            inf_frame = cv2.resize(frame, self.inference_frame_size)

            # Detect and track
            detections = self.card_detector.detect(inf_frame)
            tracked = self.card_tracker.update(detections)
            hands_info = self.hand_tracker.update(tracked)

            # Evaluate EVs
            evs = self.hand_evaluator.evaluate_hands(hands_info)

            # Update display data
            self.display.update_tracking(tracked)
            self.display.update_hands(hands_info)
            self.display.update_evaluation(evs)

            # Enqueue for display
            self._enqueue_safe(
                self.proc_queue, (inf_frame, list(detections.keys()), tracked)
            )
            self.last_inference = now

    def _display_loop(self) -> None:
        """
        Pull processed frames, annotate them, render via display, and exit on user quit input.
        """
        while self.running:
            try:
                item = self.proc_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, raw_boxes, tracked = item

            annotated = self.annotator.annotate(frame.copy(), raw_boxes, tracked)
            display_frame = cv2.resize(annotated, self.display_frame_size)
            self.display.update_frame(display_frame)

            if not self.display.handle_input():
                break

        self.running = False

    def _enqueue_safe(self, q: queue.Queue, item: Any) -> None:
        """
        Safely enqueue item into q. If the queue is full, the oldest element is discarded before enqueuing the new
        one.

        Parameters:
          q (queue.Queue): The target queue.
          item (Any): The element to enqueue.
        """
        try:
            q.put(item, timeout=0.01)
        except queue.Full:
            _ = q.get_nowait()
            q.put(item)

    def run(self) -> None:
        """
        Start capture, analysis, and display threads. Blocks all threads until completion.
        """
        logger.info("Starting AnalysisEngine")
        self.running = True

        threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._analysis_loop, daemon=True),
            threading.Thread(target=self._display_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.info("AnalysisEngine stopped")
