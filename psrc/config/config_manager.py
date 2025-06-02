from typing import Tuple, Union

import os
import yaml


class ConfigManager:
    """
    ConfigManager is responsible for loading and exposing application settings from a YAML file.
    """

    yolo_path: str
    ev_jar_path: str
    ev_class_path: str

    video_source: Union[int, str]

    inference_interval: float

    overlap_threshold: float
    iou_threshold: float
    confidence_threshold: float

    confirmation_frames: int
    removal_frames: int

    inference_frame_size: Tuple[int, int]
    annotation_frame_size: Tuple[int, int]
    window_frame_size: Tuple[int, int]
    window_name: str

    deck_count: int

    def __init__(self, config_file: str = "config.yaml") -> None:
        """
        Initialize ConfigManager by loading and parsing the given YAML configuration file.

        This implementation checks that the file exists, safely loads it, extracts the analysis_settings
        section, and sets each expected key as a typed attribute on the instance.

        Parameters:
            config_file (str): The path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
        """
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Failed to load config.yaml: " + config_file)

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        settings = config_data["analysis_settings"]

        self.video_source = settings["video_source"]

        self.yolo_path = settings["yolo_path"]
        self.ev_jar_path = settings["ev_jar_path"]
        self.ev_class_path = settings["ev_class_path"]

        self.inference_interval = settings["inference_interval"]

        self.overlap_threshold = settings["overlap_threshold"]
        self.iou_threshold = settings["iou_threshold"]
        self.confidence_threshold = settings["confidence_threshold"]

        self.confirmation_frames = settings["confirmation_frames"]
        self.removal_frames = settings["removal_frames"]

        self.inference_frame_size = tuple(settings["inference_frame_size"])
        self.annotation_frame_size = tuple(settings["annotation_frame_size"])
        self.window_frame_size = tuple(settings["window_frame_size"])
        self.window_name = settings["window_name"]

        self.deck_count = settings["deck_count"]
