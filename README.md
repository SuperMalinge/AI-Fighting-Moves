# Fighting Style AI Analyzer

An advanced AI system that learns fighting styles from video content and predicts optimal defensive moves. The system uses computer vision and deep learning to analyze combat techniques and provide real-time defense suggestions.

![Fighting Style AI Demo](demo.gif)

## Features

- Video-based fighting style analysis
- Real-time pose detection and tracking
- Deep learning model for move prediction
- User-friendly GUI interface
- Save and load trained models
- Export fighting poses for reference
- Support for multiple video formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fighting-style-ai.git

fighting-style-ai/
├── main.py         # Main application file
├── saved_poses/         # Directory for saved pose data
├── models/             # Directory for trained models
└── training_data/      # Directory for training videos

```

create virtual enviroment:
python -m venv venv

Install required packages:
pip install tensorflow opencv-python numpy moviepy mediapipe pillow

start the script:
python main.py

Use the GUI to:
Load training videos
Train the AI model
Analyze fighting sequences
Save detected poses
Export trained models
