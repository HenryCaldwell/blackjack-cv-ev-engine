# Blackjack Computer Vision Evaluation Engine

## Project Description

A real-time computer vision system for evaluating optimal blackjack decisions. This application uses YOLO for card detection, a custom tracking and hand grouping pipeline, and a Java-based expected value calculator to evaluate stand, hit, double, split, or surrender actions. Results are displayed as an annotated video alongside a Rich-powered dashboard of hand info, expected values, and deck composition.

## Table of Contents

- [Project Description](#project-description)
  - [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Usage](#usage)
  - [Run the Application](#run-the-application)
  - [Terminate the Program](#terminate-the-program)
  - [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [License](#license)

**Key Features:**

- **Card Detection:** Utilizes an Ultralytics YOLO model to detect playing cards in each video frame with high accuracy and low latency.
- **Card Tracking:** Employs a Hungarian-algorithm–based tracker to maintain consistent card tracks across frames, confirming and pruning tracks automatically.
- **Hand Grouping:** Clusters detected cards into dealer and player hands based on bounding-box overlap, scoring each hand according to blackjack rules.
- **Expected Value Computation:** Integrates a Java-based expected value calculator (via JPype) to compute expected values for stand, hit, double, split, and surrender decisions on the fly.
- **Multi-Threaded Processing Pipeline:** Separates capture, analysis, and display into dedicated threads to ensure smooth video input, uninterrupted inference, and responsive UI updates.
- **Live Annotated Display:** Renders video frames with Matplotlib overlays (bounding boxes and labels) and a Rich-powered sidebar showing hand details, expected value breakdowns, and remaining deck composition simultaneously.
- **Flexible Configuration:** All thresholds, model paths, blackjack rules, and display settings are exposed in a single YAML file, making it easy to tweak detection parameters, deck counts, and other blackjack rules without touching code.

## Installation and Setup

### Prerequisites

- **Python 3.8+**
- **Java JDK 11+**
- **Maven**
- **Pretrained YOLO weights**

### Setup Steps

1. **Clone the Repository:**

```bash
git clone https://github.com/HenryCaldwell/blackjack-cv-ev-engine.git
cd blackjack-cv-ev-engine
```

2. **Install Python Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Build the Java Expected Value Calculator:**

```bash
mvn clean package
```

4. **Verify the YOLO Weights:**

- By default, config.yaml expects the YOLO weights file at resources/detection_weights.pt.

5. **Review and Adjust Configuration:**

- Edit `config.yaml` at the project root to set:
  - Paths to the YOLO weights and video source.
  - Java EV calculator JAR and class paths.
  - Any thresholds, frame sizes, or blackjack rules as needed.

## Usage

1. **Run the Application:**

```bash
python main.py
```

- Three threads start automatically:
  - Capture thread — reads frames from webcam or video file.
  - Analysis thread — performs card detection, tracking, hand grouping, and EV evaluation.
  - Display thread — shows annotated frames in an OpenCV window plus Rich live tables.

2. **Terminate the Program:**

- Close the OpenCV window to stop all threads and exit.

3. **Troubleshooting:**

- JPype "JVM not found" errors: Ensure JAVA_HOME points to a Java 11+ installation.
- "Unable to open video source": Verify video_path in config.yaml, or set use_webcam: true with the correct webcam_index.
- YOLO model loading issues: Confirm yolo_path references a valid Ultralytics-compatible .pt file.
- Python dependency conflicts: Make sure the virtual environment is activated before running pip install -r requirements.txt.

## File Structure

```
.
├── config.yaml                     # Stores all analysis, detection, display, and blackjack settings.
├── main.py                         # Initializes components and starts the evaluation engine.
├── pom.xml                         # Configures Maven to compile and package the Java EV calculator.
├── requirements.txt                # Lists the Python dependencies for the project.
├── jsrc
│  └── evaluation
│     ├── ConfigManager.java        # Loads blackjack game settings.
│     ├── EVCalculator.java         # Calculates expected values for blackjack actions.
│     └── StateKey.java             # Represents a game state for EV caching.
├── psrc
│  ├── annotation
│  │  └── cv_annotator.py           # Draws bounding boxes and labels on frames.
│  ├── config
│  │  └── config_manager.py         # Loads analysis settings.
│  ├── core
│  │  ├── analysis_engine.py        # Runs capture, analysis, and display loops.
│  │  └── interfaces
|  |     ├── i_card_deck.py         # Defines methods for adding/removing cards in a deck.
|  |     ├── i_card_detector.py     # Defines how to detect cards in a frame.
|  |     ├── i_card_tracker.py      # Defines how to track cards across frames.
|  |     ├── i_display.py           # Defines methods for rendering frames and handling UI.
|  |     ├── i_ev_calculator.py     # Defines methods for computing blackjack EVs.
|  |     ├── i_frame_annotator.py   # Defines how to annotate video frames.
|  |     ├── i_frame_reader.py      # Defines how to read frames from a video source.
|  |     ├── i_hand_evaluator.py    # Defines how to evaluate blackjack hands.
|  |     └── i_hand_tracker.py      # Defines how to group tracked cards into hands.
│  ├── debugging
│  │  └── logger.py                 # Sets up a logger with timestamped output.
│  ├── detection
│  │  ├── card_detector.py          # Runs a YOLO model to detect cards.
│  │  ├── card_tracker.py           # Matches detections to tracks and manages track states.
│  │  └── hand_tracker.py           # Groups card tracks into blackjack hands.
│  ├── display
│  │  └── hybrid_display.py         # Shows annotated video and related tables side by side.
│  ├── evaluation
│  │  ├── card_deck.py              # Manages counts for a multi-deck blackjack deck.
│  │  ├── ev_calculator_wrapper.py  # Wraps the Java EV calculator via JPype.
│  │  └── hand_evaluator.py         # Chooses the best blackjack action based on EVs.
│  └── input
│     └── cv_video_stream.py        # Reads frames from a video file or webcam.
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
