# Public Safety System

## Overview
The **Public Safety System** is an AI-powered surveillance system designed to enhance public safety by detecting anomalies, counting people, identifying weapons, and tracking vehicles in real time using YOLO models and computer vision techniques.

## Features
- **Anomaly Detection**: Identifies unusual activities in monitored areas.
- **People Counting**: Counts the number of individuals in a frame.
- **Weapon Detection**: Detects the presence of weapons in real-time.
- **Vehicle Tracking**: Uses object tracking to monitor vehicle movement and counting.
- **Face Recognition**: Identifies specific individuals using facial detection (Helps in identifying individuals with criminal records).

## Project Structure
```
Public-Safety-System/
│── data/                  # Folder for storing reference images and videos
│── modules/               # Contains all core processing modules
│   │── anomaly.py         # Detects anomalies
│   │── people_count.py    # Counts the number of people
│   │── sort.py            # Object tracking algorithm (SORT)
│   │── weapon.py          # Detects weapons
│   │── yolo_cars.py       # Vehicle tracking
│── yolo_weights/          # YOLO model weights , will download and appear when you run programs.
│── .gitignore             # Specifies files to ignore in version control
│── README.md              # Project documentation
│── requirements.txt       # List of dependencies
```

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/rajrasane/Public-Safety-System.git
cd Public-Safety-System
```
### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```sh
python -m venv my_project
source my_project/bin/activate  # On macOS/Linux
my_project\Scripts\activate     # On Windows
```
**Or using Anaconda:**
```sh
conda create --name my_project python=3.8
conda activate my_project
```
### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Run Face Detection and Person Tracking
```sh
python modules/people_count.py
```
### Run Anomaly Detection
```sh
python modules/anomaly.py
```
### Run Weapon Detection
```sh
python modules/weapon.py
```
### Run Vehicle Tracking
```sh
python modules/yolo_cars.py
```

## Model Weights
The YOLO model weights (YOLOv8n, YOLOv8l) are automatically downloaded when running the respective modules. If not, download them manually from the official YOLO repository and place them in the `yolo_weights/` directory.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to your branch and create a pull request.