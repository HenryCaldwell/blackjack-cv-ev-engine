import cv2
import threading
import time
import queue
from typing import Any, Optional, Tuple

from psrc.core.interfaces.i_frame_annotator import IAnnotator
from psrc.core.interfaces.i_card_deck import ICardDeck
from psrc.core.interfaces.i_card_detector import ICardDetector
from psrc.core.interfaces.i_card_tracker import ICardTracker
from psrc.core.interfaces.i_hand_tracker import IHandTracker
from psrc.core.interfaces.i_hand_evaluator import IHandEvaluator
from psrc.core.interfaces.i_display import IDisplay
from psrc.core.interfaces.i_frame_reader import IVideoStreamReader

from psrc.debugging.logger import setup_logger

logger = setup_logger(__name__)


class AnalysisEngine:
    """
    Coordinates the end-to-end blackjack CV evaluation by running three worker threads. The capture thread
    continuously reads video frames and enqueues them; the analysis thread wakes at a fixed interval to dequeue
    a frame, perform card detection, tracking, hand grouping, and EV calculations, then enqueues the results;
    the display thread dequeues processed data, annotates the frame, and updates the UI. Thread-safe queues are
    used to transfer data between threads.
    """

    def __init__(
        self,
        video_reader: IVideoStreamReader,
        card_detector: ICardDetector,
        card_tracker: ICardTracker,
        hand_tracker: IHandTracker,
        deck: ICardDeck,
        hand_evaluator: IHandEvaluator,
        annotator: IAnnotator,
        display: IDisplay,
        inference_interval: float = 0.25,
        inference_frame_size: Tuple[int, int] = (1280, 720),
        annotation_frame_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        """
         Initialize the AnalysisEngine with all the components required for the video processing workflow.

        Parameters:
          video_reader (IVideoStreamReader): The video reader instance for retrieving frames.
          card_detector (ICardDetector): The card detector for identifying cards in frames.
          card_tracker (ICardTracker): The card tracker for maintaining detection continuity.
          hand_tracker (IHandTracker): The hand tracker for grouping cards and computing hand scores.
          deck (ICardDeck): The card deck for tracking remaining deck composition.
          hand_evaluator (IHandEvaluator): The hand evaluator for selecting optimal actions.
          annotator (IAnnotator): The annotator for drawing bounding boxes and hand details on frames.
          display (IDisplay): The display interface for showing annotated frames and handling user input.

          inference_interval (float): The minimum time (in seconds) between inference steps.
          inference_frame_size (Tuple[int, int]): The resolution for inference processing.
          annotation_frame_size (Tuple[int, int]): The resolution for display output.
        """
        # Core components
        self.video_reader = video_reader
        self.card_detector = card_detector
        self.card_tracker = card_tracker
        self.hand_tracker = hand_tracker
        self.deck = deck
        self.hand_evaluator = hand_evaluator
        self.annotator = annotator
        self.display = display

        # Timing & sizes
        self.inference_interval = inference_interval
        self.inference_frame_size = inference_frame_size
        self.annotation_frame_size = annotation_frame_size

        # Queues for thread data
        self.frame_queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=1)
        self.data_queue: queue.Queue[Optional[Tuple]] = queue.Queue(maxsize=1)

        # Control
        self.running = False
        self.last_inference = time.monotonic() - inference_interval

    def _capture_loop(self) -> None:
        """
        Continuously read frames from the video source at its native FPS and enqueue the latest frame.
        """
        logger.info("Starting Capture Thread")

        fps = self.video_reader.get_fps()
        period = 1.0 / fps

        while self.running:
            start = time.monotonic()
            frame = self.video_reader.read_frame()

            if frame is None:
                self.running = False
                break

            self._enqueue_safe(self.frame_queue, frame)
            time.sleep(max(0.0, period - (time.monotonic() - start)))

        logger.info("Capture Thread Stopped")

    def _analysis_loop(self) -> None:
        """
        At fixed intervals, pull the latest raw frame, run detection, tracking, hand grouping, and EV evaluation,
        then enqueue the processed frame and metadata for display.
        """
        logger.info("Starting Analysis Thread")

        while self.running:
            now = time.monotonic()

            if now - self.last_inference < self.inference_interval:
                continue

            frame = self._dequeue_safe(self.frame_queue)

            if frame is None:
                continue

            inference_frame = cv2.resize(frame, self.inference_frame_size)

            detections = self.card_detector.detect(inference_frame)
            tracks = self.card_tracker.update(detections)
            hands = self.hand_tracker.update(tracks)
            evals = self.hand_evaluator.evaluate_hands(hands)
            deck = self.deck.cards

            self._enqueue_safe(
                self.data_queue,
                (
                    inference_frame,
                    detections,
                    tracks,
                    hands,
                    evals,
                    deck,
                ),
            )

            self.last_inference = now

        logger.info("Analysis Thread Stopped")

    def _display_loop(self) -> None:
        """
        Pull processed frames, annotate them, render via display, and exit on user input.
        """
        logger.info("Starting Display Thread")

        while self.running:
            if not self.display.process_events():
                self.running = False
                break

            bundle = self._dequeue_safe(self.data_queue)

            if bundle is None:
                continue

            (
                frame,
                detections,
                tracks,
                hands,
                evals,
                deck,
            ) = bundle

            h_inf, w_inf = frame.shape[:2]
            w_disp, h_disp = self.annotation_frame_size
            scale_x, scale_y = w_disp / w_inf, h_disp / h_inf

            resized_frame = cv2.resize(frame, (w_disp, h_disp))

            scaled_detections = {}
            for raw_bbox, det_meta in detections.items():
                x1, y1, x2, y2 = map(int, raw_bbox)
                scaled_box = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )
                new_meta = det_meta.copy()
                scaled_detections[scaled_box] = new_meta

            scaled_tracks = {}
            for track_id, tr_meta in tracks.items():
                raw_bbox = tr_meta["bbox"]
                x1, y1, x2, y2 = map(int, raw_bbox)
                scaled_box = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )
                new_meta = tr_meta.copy()
                new_meta["bbox"] = scaled_box
                scaled_tracks[track_id] = new_meta

            annotated_frame = self.annotator.annotate(
                resized_frame, scaled_detections, scaled_tracks
            )

            self.display.update(
                frame=annotated_frame,
                tracks=tracks,
                hands=hands,
                evals=evals,
                deck=deck,
            )

        logger.info("Display Thread Stopped")

    def _enqueue_safe(self, q: queue.Queue, item: Any) -> None:
        """
        Safely enqueue item into q. If the queue is full, the oldest element is discarded before enqueuing the
        new one.

        Parameters:
          q (queue.Queue): The target queue.
          item (Any): The element to enqueue.
        """
        try:
            q.put(item, timeout=0.01)
        except queue.Full:
            _ = q.get_nowait()
            q.put(item)

    def _dequeue_safe(self, q: queue.Queue) -> Any:
        """
        Safely dequeue an item from q. If the queue is empty, None is returned.

        Parameters:
          q (queue.Queue): The target queue.
        """
        try:
            return q.get(timeout=0.01)
        except queue.Empty:
            return None

    def start(self) -> None:
        """
        Start capture, analysis, and display threads. Blocks all threads until completion.
        """
        logger.info("Starting AnalysisEngine")

        self.running = True

        threads = [
            threading.Thread(target=self._capture_loop),
            threading.Thread(target=self._analysis_loop),
            threading.Thread(target=self._display_loop),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.info("AnalysisEngine Stopped")
