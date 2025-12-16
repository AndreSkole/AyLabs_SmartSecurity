# AyLabs Smart Security System


AyLabs Smart Security is a real-time face recognition system built with Python, Flask, and OpenCV. It provides a web-based interface for monitoring a live camera feed, detecting known and unknown individuals, registering new faces, and viewing a complete history of all detections.

## Features

*   **Live Face Recognition**: Real-time video stream in the browser with detected faces highlighted.
*   **Known & Unknown Detection**: Recognizes registered individuals and flags unknown faces.
*   **Web-Based Registration**: A user-friendly interface to view images of detected unknown people and register them with a name.
*   **Model Training**: Easily retrain the face recognition model from the web interface after adding new people.
*   **Detection History**: A comprehensive, filterable log of all detections (both known and unknown), complete with images, names, and timestamps.
*   **System Dashboard**: At-a-glance statistics showing the number of known persons, pending unknown faces, and total detections.
*   **Automated Image Capture**: Automatically saves images of unknown faces for later registration and logs all detections for historical review.

## Technology Stack

*   **Backend**: Python, Flask
*   **Computer Vision**: OpenCV, NumPy
*   **Frontend**: HTML, CSS, JavaScript

## File Structure

The repository has a simple and organized structure:

```
.
├── app.py                  # Main Flask application with all backend logic
├── templates/
│   └── index.html          # Single-page HTML interface for the application
├── database/               # (Auto-created) Stores images for training the model
├── detections/             # (Auto-created) Stores an image for every detection event
├── trainer.yml             # (Auto-created) The trained face recognition model file
└── system.log              # (Auto-created) Log file for server events
```

## Setup and Installation

Follow these steps to get the application running on your local machine.

**Prerequisites:**
*   Python 3.x
*   A webcam connected to your computer.

**1. Clone the repository:**
```bash
git clone https://github.com/AndreSkole/AyLabs_SmartSecurity.git
cd AyLabs_SmartSecurity
```

**2. Create a virtual environment (recommended):**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required Python packages:**
The project relies on OpenCV, and it's crucial to install the `contrib` package which includes the face recognition module.
```bash
pip install flask numpy opencv-python opencv-contrib-python
```

**4. Run the application:**
```bash
python app.py
```
The server will start, and you will see log messages in your terminal indicating that the system is running.

## How to Use

1.  **Start the System**: Run `python app.py` in your terminal.
2.  **Access the Web Interface**: Open your web browser and navigate to `http://127.0.0.1:5000`.
3.  **Live Monitoring**: The "Live Kamera" tab shows the real-time feed from your webcam. Any faces detected will be framed. Initially, all faces will be marked as "Unknown". The system will automatically save a snapshot of each unique unknown face to the `database/` directory.
4.  **Register New Faces**:
    *   Navigate to the "Registrering" (Registration) tab.
    *   Here you will see a gallery of captured "Unknown" faces.
    *   Enter a name for each person in the text box below their picture and click the "✅ Registrer" button.
    *   It is recommended to have multiple images (5-10) of a person from different angles for better accuracy. Let the system capture them over time or add them manually to the `database/` folder with the format `{Name}_{timestamp}.jpg`.
5.  **Train the Model**:
    *   After registering one or more new people, click the "🎓 Tren Modell" (Train Model) button.
    *   The system will process all the registered images in the `database/` directory and create/update the `trainer.yml` file.
    *   A success message will confirm that the model has been trained. The "Modell Status" on the dashboard will change to `✅`.
6.  **View Recognized Faces**: Return to the "Live Kamera" tab. The system will now recognize and label the people you have trained it on.
7.  **Review History**:
    *   Navigate to the "Historikk" (History) tab.
    *   This page displays a reverse-chronological view of every face detection event.
    *   Use the filter controls at the top to search for specific people, dates, or filter by known/unknown detections.
