# Blackjack Computer Vision Evaluation Engine

## Project Description

The Blackjack Computer Vision Evaluation Engine is a modular application that integrates real-time video processing with blackjack game analysis. It uses computer vision techniques to detect and track playing cards in a video stream, groups them into blackjack hands, and calculates the expected value (EV) of various blackjack actions (stand, hit, double, and split).

**Key Features:**

- **Real-Time Card Detection:** Uses a YOLO-based model with OpenCV to detect playing cards from video frames.
- **Card Tracking and Hand Grouping:** Tracks detections across frames and groups them into dealer and player hands using IoU algorithms.
- **Blackjack EV Calculations:** Computes the expected value of different actions using a recursive, memoized Java EV calculator.
- **Seamless Integration:** Combines Python’s rapid prototyping and computer vision capabilities with Java’s performance for EV computations.
- **Modular Design:** Organized into clear components that adhere to defined interfaces for easy maintenance and future expansion.

**Technologies Used:**

- **Python:** For video processing, computer vision (OpenCV), detection (YOLO via ultralytics), and overall pipeline orchestration.
- **Java:** For the EV calculation engine, using Maven for build management.
- **JPype:** To bridge Python and Java, allowing the application to leverage Java’s performance within a Python environment.

## Table of Contents

1. [Project Description](#project-description)
2. [Installation and Setup](#installation-and-setup)
3. [Usage](#usage)
4. [File Structure](#file-structure)

## Installation and Setup

### Prerequisites

- **Python 3.8+**
- **Java JDK 11+**
- **Maven**
- **Required Python Packages:**
  - OpenCV (`opencv-python`)
  - PyYAML
  - JPype1
  - ultralytics (for the YOLO model)

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

3. **Build the Java EV Calculator:**

```bash
cd jsrc/evaluation
mvn clean package
```

This creates a JAR (e.g., `target/blackjack-ev-calculator-1.0.0.jar`).

4. **Configure the Application:**

- Edit `config.yaml` at the project root to set:
  - Paths to the YOLO weights and test video.
  - Inference settings.
  - Blackjack rules and deck settings.
  - Java EV calculator JAR and class paths.

## Usage

### Running the Application

1. **Ensure Configuration is Correct:**  
   Verify that `config.yaml` contains the correct file paths and settings for your environment.

2. **Start the Application:**

```bash
python main.py
```

3. **Interact with the Application:**
   - An OpenCV window will open showing video frames annotated with bounding boxes and EV information.
   - The console will display logging information regarding card detection, tracking, and evaluation.
   - Press `q` on the display window to quit the application.

### Additional Notes

- **Video Source:**  
  Toggle between using a video file and a webcam by setting `use_webcam` in `config.yaml`.
- **Debugging and Logging:**  
  The project uses a custom logger (`psrc/debugging/logger.py`) to record runtime events, which helps in troubleshooting and further development.

## File Structure

```
.
├── config.yaml                     # Contains the main configuration settings for detection, display, and blackjack rules
├── main.py                         # The main entry point for the application
├── pom.xml                         # Maven configuration for building the Java EV calculator
├── README.md                       # Contains project documentation and guidelines
├── requirements.txt                # Lists the required Python dependencies
├── jsrc
│  └── evaluation
│     ├── ConfigManager.java        # Loads game settings from config.yaml using SnakeYAML
│     ├── EVCalculator.java         # Implements recursive expected value (EV) calculations for blackjack actions
│     └── StateKey.java             # Generates unique keys for memoization in EV calculations
├── psrc
│  ├── annotation
│  │  └── cv_annotator.py           # Annotates video frames with detected card and hand details using OpenCV
│  ├── config
│  │  └── config_manager.py         # Loads and parses the YAML configuration for the Python modules
│  ├── core
│  │  ├── analysis_engine.py        # Orchestrates the complete video processing pipeline including detection, tracking, EV calculation, and display
│  │  └── interfaces                # Contains abstract interfaces defining contracts for various modules (e.g., detector, tracker, display)
│  ├── debugging
│  │  └── logger.py                 # Configures and initializes the custom logger
│  ├── detection
│  │  ├── card_detector.py          # Detects cards in video frames using a YOLO-based model
│  │  ├── card_tracker.py           # Tracks card detections across frames to maintain continuity
│  │  ├── hand_tracker.py           # Groups detected cards into dealer and player hands and computes their scores
│  │  └── detection_utils.py        # Contains utility functions for computing bounding box overlaps and grouping
│  ├── evaluation
│  │  ├── card_deck.py              # Manages the blackjack card deck by adding or removing cards
│  │  ├── ev_calculator_wrapper.py  # Provides a Python wrapper for the Java EV calculator using JPype
│  │  └── java_conversion_utils.py  # Converts Python data structures to Java-compatible types for EV calculations
│  ├── ui
│  │  └── cv_display.py             # Displays video frames and manages user input using OpenCV
│  └── video
│     └── cv_video_stream.py        # Captures video frames from a file or webcam using OpenCV
```

## Future Improvements

- **User Interface Enhancements:** Develop a more robust GUI for better visualization and user interaction.
- **Extended Game Logic:** Incorporate additional blackjack strategies and rules to further enhance EV analysis.
- **Performance Optimization:** Refine the recursive EV calculation and improve overall processing speed.
- **Improved Error Handling:** Enhance logging and add more comprehensive error handling to manage real-time processing issues.
