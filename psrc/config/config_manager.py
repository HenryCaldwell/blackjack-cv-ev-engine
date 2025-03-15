import os
import yaml

from typing import Tuple

class ConfigManager:
  yolo_path: str
  video_path: str

  use_webcam: bool
  webcam_index: int

  inference_interval: float
  inference_frame_size: Tuple[int, int]

  overlap_threshold: float
  inference_overlap_threshold: float
  confidence_threshold: float

  confirmation_frames: int
  disappear_frames: int

  display_frame_size: Tuple[int, int]

  deck_size: int

  def __init__(self, config_file: str = "config.yaml") -> None:
    if not os.path.isfile(config_file):
      raise FileNotFoundError("Failed to load config.yaml: " + config_file)
    with open(config_file, "r") as f:
      config_data = yaml.safe_load(f)
    
    settings = config_data["analysis_settings"]

    self.yolo_path = settings["yolo_path"]
    self.video_path = settings["video_path"]

    self.use_webcam = settings["use_webcam"]
    self.webcam_index = settings["webcam_index"]

    self.inference_interval = settings["inference_interval"]
    self.inference_frame_size = tuple(settings["inference_frame_size"])

    self.overlap_threshold = settings["overlap_threshold"]
    self.inference_overlap_threshold = settings["inference_overlap_threshold"]
    self.confidence_threshold = settings["confidence_threshold"]

    self.confirmation_frames = settings["confirmation_frames"]
    self.disappear_frames = settings["disappear_frames"]

    self.ev_jar_path = settings["ev_jar_path"]
    self.ev_class_path = settings["ev_class_path"]

    self.display_frame_size = tuple(settings["display_frame_size"])
    self.window_name = settings["window_name"]

    self.player_color = tuple(settings["player_color"])
    self.dealer_color = tuple(settings["dealer_color"])
    self.default_color = tuple(settings["default_color"])

    self.font_scale = settings["font_scale"]
    self.thickness = settings["thickness"]

    self.deck_size = settings["deck_size"]