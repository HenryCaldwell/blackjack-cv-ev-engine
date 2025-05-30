from typing import Any, Dict

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

from psrc.core.interfaces.i_frame_annotator import IFrameAnnotator

matplotlib.use("Agg")


class MPLAnnotator(IFrameAnnotator):
    """
    MPLAnnotator is an implementation of the IFrameAnnotator interface.

    This implementation uses Matplotlib’s Agg backend to draw detection and track overlays on frames. Bounding
    boxes are colored by track state, and labels are rendered with proportional thickness and font size.

    Attributes:
        _BOX_THICKNESS_RATIO (float): The relative thickness of box edges versus frame height.
        _FONT_SIZE_RATIO (float): The relative font size versus frame height.
        _CONFIRMED_COLOR (Tuple[float, float, float]): The RGB color for confirmed tracks.
        _TENTATIVE_COLOR (Tuple[float, float, float]): The RGB color for tentative tracks.
    """

    _BOX_THICKNESS_RATIO = 0.003
    _FONT_SIZE_RATIO = 0.01

    _CONFIRMED_COLOR = (0, 1, 0)
    _TENTATIVE_COLOR = (1, 0, 0)

    def annotate(
        self,
        frame: np.ndarray,
        detections: Dict[tuple, Dict[str, Any]],
        tracks: Dict[int, Dict[str, Any]],
    ) -> np.ndarray:
        """
        Annotate a frame with detection and tracking information.

        This implementation converts the OpenCV BGR frame to an RGB Matplotlib figure, for each confirmed
        track draws a colored Rectangle patch and italic bold text showing track ID and label, and converts
        back to BGR for OpenCV compatibility.

        Parameters:
            frame (Any): The frame to annotate.
            detections (Dict[Tuple, Dict[str, Any]]): A mapping of bounding box coordinates to their detection
            information.
            tracks (Dict[int, Dict[str, Any]]): A mapping of track IDs to their tracking information.

        Returns:
            Any: The annotated frame.
        """
        h, w = frame.shape[:2]

        box_thickness = max(1, self._BOX_THICKNESS_RATIO * h)
        font_size = max(1, self._FONT_SIZE_RATIO * h)

        fig = plt.figure(frameon=False, dpi=100)
        fig.set_size_inches(w / fig.dpi, h / fig.dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")

        for track_id, track_meta in tracks.items():
            bbox = track_meta["bbox"]

            if bbox not in detections:
                continue

            label = track_meta.get("label", "N/A")
            state = track_meta.get("state", 0)

            x1, y1, x2, y2 = map(int, bbox)
            text = f"ID: {track_id}, VAL: {label}"
            box_color = self._CONFIRMED_COLOR if state == 1 else self._TENTATIVE_COLOR
            text_color = tuple(c * 0.75 for c in box_color)

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=box_thickness,
                edgecolor=box_color,
                facecolor="none",
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - box_thickness * 2,
                text,
                fontsize=font_size,
                color=text_color,
                fontstyle="italic",
                weight="bold",
            )

        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        rgba = buf.reshape((int(h), int(w), 4))
        rgb = rgba[..., :3]
        plt.close(fig)

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
