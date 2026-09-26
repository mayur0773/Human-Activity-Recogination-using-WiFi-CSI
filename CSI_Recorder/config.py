# ==========================================
# WiFi CSI Recorder Configuration
# ==========================================

# Serial Port
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# Recording
RECORD_DURATION = 15          # seconds
NUMBER_OF_SAMPLES = 3

# Dataset Path
DATASET_PATH = r"C:\CSI_Project\Practice_Dataset"

# Activities
ACTIVITIES = [
    "Empty",
    "Standing",
    "Sitting",
    "Walking"
]

# CSV Prefix
FILE_PREFIX = {
    "Empty": "Empty",
    "Standing": "Standing",
    "Sitting": "Sitting",
    "Walking": "Walking"
}