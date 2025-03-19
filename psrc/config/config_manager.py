import os
import yaml

from typing import Tuple

class ConfigManager:
  """
  ConfigManager loads and parses configuration settings the centralized YAML file.

  This class reads configuration parameters such as paths, inference settings, display options, and deck size
  from a YAML configuration file (default "config.yaml"). The loaded configuration is stored in corresponding
  instance attributes for use throughout the application.
  """
  
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
    """
    Initialize the ConfigManager by loading settings from a YAML configuration file.

    Parameters:
      config_file (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Raises:
      FileNotFoundError: If the specified configuration file does not exist.
    """
    # Verify that the configuration file exists
    if not os.path.isfile(config_file):
      raise FileNotFoundError("Failed to load config.yaml: " + config_file)
    
    # Open and parse the YAML configuration file
    with open(config_file, "r") as f:
      config_data = yaml.safe_load(f)
    
    # Extract the analysis settings from the configuration data
    settings = config_data["analysis_settings"]

    # Set YOLO and video paths
    self.yolo_path = settings["yolo_path"]
    self.video_path = settings["video_path"]

    # Set expected value calculator JAR and class paths
    self.ev_jar_path = settings["ev_jar_path"]
    self.ev_class_path = settings["ev_class_path"]

    # Set webcam usage and index
    self.use_webcam = settings["use_webcam"]
    self.webcam_index = settings["webcam_index"]

    # Set inference parameters
    self.inference_interval = settings["inference_interval"]
    self.inference_frame_size = tuple(settings["inference_frame_size"])

    # Set thresholds for inference
    self.overlap_threshold = settings["overlap_threshold"]
    self.iou_threshold = settings["iou_threshold"]
    self.confidence_threshold = settings["confidence_threshold"]

    # Set frame count parameters for detection
    self.confirmation_frames = settings["confirmation_frames"]
    self.removal_frames = settings["removal_frames"]

    # Set display parameters
    self.display_frame_size = tuple(settings["display_frame_size"])
    self.window_name = settings["window_name"]

    # Set text settings
    self.player_color = tuple(settings["player_color"])
    self.dealer_color = tuple(settings["dealer_color"])
    self.default_color = tuple(settings["default_color"])
    self.font_scale = settings["font_scale"]
    self.thickness = settings["thickness"]

    # Set the deck count parameter
    self.deck_count = settings["deck_count"]