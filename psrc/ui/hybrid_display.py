import cv2
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from psrc.core.interfaces.i_display import IDisplay


class HybridDisplay(IDisplay):
    """
    HybridDisplay implements the IDisplay interface for displaying frames and evaluation data.

    It displays live video in an OpenCV window and renders three console tables using Rich.
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

    def __init__(self, window_name: str = "Blackjack CV"):
        """
        Initialize the HybridDisplay with an OpenCV window and empty state.

        Parameters:
          window_name (str): The name of the OpenCV window.
        """
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

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
        Update any subset of the display’s state under the thread lock.

        Parameters:
          frame (Any, optional): The latest video frame to display.
          detections (Dict[tuple, Dict[str, Any]], optional): Raw detection boxes mapped to detection info.
          tracks (Dict[int, Dict[str, Any]], optional): Tracked card IDs mapped to bbox/label/state.
          hands (Dict[str, Any], optional): Grouped hand information.
          evals (Dict[str, Any], optional): Expected value results and best actions for each hand.
          deck (Dict[int, int], optional): Current deck composition as card label count.
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
        Pump OpenCV GUI events and check if the window is still visible.

        Returns:
          bool: False if the window has been closed (signal to stop), True otherwise.
        """
        cv2.waitKey(1)
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    def release(self) -> None:
        """
        Release the display and any associated resources.
        """
        cv2.destroyAllWindows()

    def _build_table(self, title: str, columns: List[Tuple[str, str, int]]) -> Table:
        """
        Create a Rich Table with the given title and column definitions.

        Parameters:
          title (str): Table title shown at the top.
          columns (List[Tuple[str,str,int]]): List of (header, justify, min_width).
        """
        tbl = Table(title=title, title_justify="center")
        for header, justify, width in columns:
            tbl.add_column(header, justify=justify, min_width=width)
        return tbl

    def _make_hand_table(self) -> Table:
        """
        Build and populate the hand information table:
          Hand ID | Cards (formatted) | Score (colored green/red).
        """
        tbl = self._build_table("HAND INFORMATION", self.HAND_COLUMNS)
        for hid, info in self._hands.items():
            cards = [
                self.RANKS[int(card)] if 0 <= int(card) < len(self.RANKS) else str(card)
                for card in info.get("cards", [])
            ]
            display_cards = ", ".join(cards)
            score = info.get("score", 0)
            color = "green" if score <= 21 else "red"
            score_str = f"[{color}]{score}[/{color}]"
            tbl.add_row(hid, display_cards, score_str)
        return tbl

    def _make_ev_table(self) -> Table:
        """
        Build and populate the EV table:
          Hand | STD | HIT | DBL | SPL | SUR | BEST
        with values colored by positive/negative.
        """
        tbl = self._build_table("EXPECTED VALUE INFORMATION", self.EV_COLUMNS)
        for hand, res in self._evals.items():
            evs = res.get("evs", {})
            best = res.get("best_action", "")

            def fmt(key: str) -> str:
                v = evs.get(key, 0.0)
                col = "green" if v >= 0 else "red"
                return f"[{col}]{v:.2f}[/{col}]"

            tbl.add_row(
                hand,
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
        Build and populate the deck composition table:
          Card label | Remaining count.
        """
        tbl = self._build_table("DECK COMPOSITION", self.DECK_COLUMNS)
        for label in sorted(self._deck.keys()):
            display = "A" if label == 0 else str(label + 1)
            tbl.add_row(display, str(self._deck[label]))
        return tbl

    def start(self) -> None:
        """
        Main display loop (must run on the main thread). Continues until the window is closed.
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
                    cv2.imshow(self.window_name, frame)

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
