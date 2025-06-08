from typing import Any, Dict, List, Optional, Tuple

from threading import Lock

import cv2
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from psrc.core.interfaces.i_display import IDisplay


class HybridDisplay(IDisplay):
    """
    HybridDisplay is an implementation of the IDisplay interface.

    This implementation uses OpenCV to render video frames in a window and Rich Live/Table to display hand
    info, expected-value recommendations, and deck composition alongside the video. Internal state updates are
    thread-safe via a Lock.
    """

    RANKS: List[str] = [
        "A",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "J",
        "Q",
        "K",
    ]
    EV_COLUMNS: List[Tuple[str, str, int]] = [
        ("HAND", "center", 8),
        ("STD", "center", 5),
        ("HIT", "center", 5),
        ("DBL", "center", 5),
        ("SPL", "center", 5),
        ("SUR", "center", 5),
        ("BEST", "center", 9),
    ]
    HAND_COLUMNS: List[Tuple[str, str, int]] = [
        ("HAND", "center", 8),
        ("CARDS", "left", 37),
        ("SCORE", "center", 9),
    ]
    DECK_COLUMNS: List[Tuple[str, str, int]] = [
        ("CARD", "center", 8),
        ("COUNT", "center", 49),
    ]

    def __init__(
        self,
        window_name: str = "Blackjack CV EV Engine",
        window_frame_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        """
        Initialize HybridDisplay with an OpenCV window and Rich console.

        Parameters:
            window_name (str): The title for the OpenCV window.
            window_frame_size (Tuple[int,int]): The width and height for display frames.
        """
        self.window_name = window_name
        self.w, self.h = window_frame_size

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.window_name, self.w, self.h)

        self._lock = Lock()
        self._frame: Optional[Any] = None
        self._detections: Dict[tuple, Any] = {}
        self._tracks: Dict[int, Any] = {}
        self._hands: Dict[str, Any] = {}
        self._evals: Dict[str, Any] = {}
        self._deck: Dict[int, int] = {}

        self.console = Console()

    def update(
        self,
        *,
        frame: Any = None,
        detections: Dict[tuple, Dict[str, Any]] = None,
        tracks: Dict[int, Dict[str, Any]] = None,
        hands: Dict[str, Any] = None,
        evals: Dict[str, Any] = None,
        deck: Dict[int, int] = None,
    ) -> None:
        """
        Update any subset of the display state.

        This implementation acquires a lock and updates any provided subset of frame, detections, tracks,
        hands, evals, and deck.

        Parameters:
            frame (Any, optional): The frame for display.
            detections (Dict[Tuple, Dict[str, Any]], optional): A mapping of bounding box coordinates to their
            detection information.
            tracks (Dict[int, Dict[str, Any]], optional): A mapping of track IDs to their tracking information.
            hands (Dict[str, Dict[str, Any]], optional): A mapping of hand IDs to their hand information.
            evals (Dict[str, Dict[str, Any]], optional): A mapping of hand IDs to their evaluation information.
            deck (Dict[int, int], optional): A mapping of card label to count.
        """
        with self._lock:
            if frame is not None:
                self._frame = frame.copy()
            if detections is not None:
                self._detections = detections.copy()
            if tracks is not None:
                self._tracks = tracks.copy()
            if hands is not None:
                self._hands = hands.copy()
            if evals is not None:
                self._evals = evals.copy()
            if deck is not None:
                self._deck = deck.copy()

    def process_events(self) -> bool:
        """
        Process UI events and determine whether the display should continue running.

        This implementation checks if the OpenCV window is still visible.

        Returns:
            bool: True to keep running, False otherwise.
        """
        cv2.waitKey(1)
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    def release(self) -> None:
        """
        Release all display resources and perform cleanup.

        This implementation destroys all OpenCV windows.
        """
        cv2.destroyAllWindows()

    def _build_table(self, title: str, columns: List[Tuple[str, str, int]]) -> Table:
        """
        Build a Rich Table with the given headers.

        Parameters:
            title (str): Table title.
            columns (List[Tuple[str, str, int]]): List of header specs.

        Returns:
            Table: A Rich Table ready for rows.
        """
        tbl = Table(title=title, title_justify="center")

        for header, justify, width in columns:
            tbl.add_column(header, justify=justify, min_width=width)

        return tbl

    def _make_hand_table(self) -> Table:
        """
        Create the hand-info table for Rich display.

        Returns:
            Table: A Rich Table with one row per hand, showing ID, cards, and colored score.
        """
        tbl = self._build_table("HAND INFORMATION", self.HAND_COLUMNS)

        for hid, info in self._hands.items():
            if hid == 0:
                display_id = "Dealer"
            else:
                display_id = f"Player {hid}"

            cards = [
                self.RANKS[int(card)] if 0 <= int(card) < len(self.RANKS) else str(card)
                for card in info.get("cards", [])
            ]
            display_cards = ", ".join(cards)
            score = info.get("score", 0)
            color = "green" if score <= 21 else "red"
            score_str = f"[{color}]{score}[/{color}]"
            tbl.add_row(display_id, display_cards, score_str)

        return tbl

    def _make_ev_table(self) -> Table:
        """
        Create the expected-value table for Rich display.

        Returns:
            Table: A Rich Table with one row per hand, containing EV columns and best action.
        """
        tbl = self._build_table("EXPECTED VALUE INFORMATION", self.EV_COLUMNS)

        for hid, res in self._evals.items():
            display_id = f"Player {hid}"
            evs = res.get("evs", {})
            best = res.get("best_action", "")

            def fmt(key: str) -> str:
                v = evs.get(key, 0.0)
                col = "green" if v >= 0 else "red"
                return f"[{col}]{v:.2f}[/{col}]"

            tbl.add_row(
                display_id,
                fmt("stand"),
                fmt("hit"),
                fmt("double"),
                fmt("split"),
                fmt("surrender"),
                f"[bold yellow]{best}[/bold yellow]",
            )

        return tbl

    def _make_deck_table(self) -> Table:
        """
        Create the deck-composition table for Rich display.

        Returns:
            Table: A Rich Table listing each card label and its remaining count.
        """
        tbl = self._build_table("DECK COMPOSITION", self.DECK_COLUMNS)

        for label in sorted(self._deck.keys()):
            display = "A" if label == 0 else str(label + 1)
            tbl.add_row(display, str(self._deck[label]))

        return tbl

    def start(self) -> None:
        """
        Run the live display loop until the window closes.

        This method opens a Rich Live context, resizes and shows frames when updated, updates Rich tables
        whenever state is updated.

        Returns:
            None
        """
        prev_hands, prev_evals, prev_deck = None, None, None

        with Live(console=self.console, screen=False, auto_refresh=True) as live:
            while True:
                with self._lock:
                    frame = self._frame
                    hands = self._hands.copy()
                    evs = self._evals.copy()
                    deck = self._deck.copy()

                if frame is not None:
                    display_frame = cv2.resize(frame, (self.w, self.h))
                    cv2.imshow(self.window_name, display_frame)

                if not self.process_events():
                    break

                if (hands, evs, deck) != (prev_hands, prev_evals, prev_deck):
                    tbls = Group(
                        self._make_hand_table(),
                        self._make_ev_table(),
                        self._make_deck_table(),
                    )
                    live.update(tbls)
                    prev_hands, prev_evals, prev_deck = hands, evs, deck
