import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Any, Dict

from psrc.core.interfaces.i_frame_annotator import IFrameAnnotator


class MPLAnnotator(IFrameAnnotator):
    """
    MPLAnnotator implements the IAnnotator interface for rendering tracked bounding boxes and labels onto video
    frames.

    Attributes:
      BOX_THICKNESS_RATIO (float): Relative thickness of box edges versus frame height.
      FONT_SIZE_RATIO   (float): Relative font size versus frame height.
      COLOR_CONFIRMED   (Tuple[float, float, float]): RGB color for confirmed tracks.
      COLOR_TENTATIVE   (Tuple[float, float, float]): RGB color for tentative tracks.
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
        Overlay tracked bounding boxes and labels onto a video frame.

        Creates an off‐screen Matplotlib canvas matching the input frame size, draws each track’s box and label
        (only if the box appears in `detections`), and returns the result as a BGR NumPy array.

        Parameters:
          frame (np.ndarray): The image frame to annotate.
          detections (Dict[Tuple[int, int, int, int], Dict[str, Any]]): A dictionary mapping detection boxes to
          detection metadata.
          tracks (Dict[int, Dict[str, Any]]): A dictionary mapping track numbers to card information
          (e.g., bbox, label, state).

        Returns:
          np.ndarray: Annotated image in BGR format with boxes and labels drawn.
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
