import csv
import os
import time
from datetime import datetime
from serial_manager import connect_serial

# Session ID: shared by every sample recorded in this single
# run of the script. Used later to group entire sessions
# together for train/test splitting (GroupKFold), so the model
# is never tested on the same sitting it was trained on.
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M")


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_FOLDER = r"C:\CSI_Project\Dataset"

ACTIVITY_DURATION = {
    "Empty": 15,
    "Standing": 15,
    "Sitting": 15,
    "Walking": 15,
    "Breathing": 60
}

BREAK_TIME = 5

activities = {
    "1": "Empty",
    "2": "Standing",
    "3": "Sitting",
    "4": "Walking",
    "5": "Breathing"
}


# ==========================================================
# HEADER
# ==========================================================

print("=" * 45)
print("         WiFi CSI Recorder")
print("=" * 45)


# ==========================================================
# ACTIVITY SELECTION
# ==========================================================

print("\nSelect Activity\n")

for key, value in activities.items():
    print(f"{key}. {value}")

choice = input("\nEnter choice: ").strip()

if choice not in activities:
    print("Invalid choice!")
    exit()

activity = activities[choice]

DURATION = ACTIVITY_DURATION[activity]


# ==========================================================
# NUMBER OF SAMPLES
# ==========================================================

try:
    num_samples = int(
        input("Enter number of samples to record: ")
    )
except ValueError:
    print("Invalid number of samples!")
    exit()


# ==========================================================
# FOLDER SETUP
# ==========================================================

folder = os.path.join(
    BASE_FOLDER,
    activity
)

os.makedirs(
    folder,
    exist_ok=True
)


# ==========================================================
# FILE NUMBERING
# ==========================================================

existing = [
    f for f in os.listdir(folder)
    if f.endswith(".csv")
]

next_number = len(existing) + 1


# ==========================================================
# INFORMATION
# ==========================================================

print(f"\nSelected Activity : {activity}")
print(f"Samples to Record : {num_samples}")
print(f"Recording Time    : {DURATION} seconds")


# ==========================================================
# BREATHING INSTRUCTIONS
# ==========================================================

if activity == "Breathing":

    print("\n========== BREATHING INSTRUCTIONS ==========")
    print("1. Sit comfortably.")
    print("2. Stay completely still.")
    print("3. Do NOT move your hands or head.")
    print("4. Breathe normally.")
    print("5. Record for the full 60 seconds.")
    print("============================================\n")


# ==========================================================
# INITIAL COUNTDOWN
# ==========================================================

print("\nRecording will start in 6 seconds...")

for i in range(6, 0, -1):
    print(f"Starting in {i}...", end="\r")
    time.sleep(1)

print("\n")


# ==========================================================
# RECORDING LOOP
# ==========================================================

for sample in range(num_samples):

    filename = (
        f"{activity}_{SESSION_ID}_{next_number + sample:03d}.csv"
    )

    filepath = os.path.join(
        folder,
        filename
    )

    print("=" * 50)
    print(
        f"Recording Sample "
        f"{sample + 1}/{num_samples}"
    )
    print("=" * 50)

    # ------------------------------------------------------
    # CONNECT ESP32
    # ------------------------------------------------------

    ser = connect_serial()

    # Prevent readline() from blocking for too long
    ser.timeout = 0.1

    # Clear old data already waiting in the serial buffer
    ser.reset_input_buffer()

    # Give serial connection a moment to stabilize
    time.sleep(1)

    # ------------------------------------------------------
    # START RECORDING
    # ------------------------------------------------------

    start = time.time()

    packet_count = 0

    last_second = DURATION

    with open(
        filepath,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Packet_No",
            "CSI_Data"
        ])

        # --------------------------------------------------
        # RECORD CSI
        # --------------------------------------------------

        while True:

            elapsed = time.time() - start

            if elapsed >= DURATION:
                break

            remaining = DURATION - int(elapsed)

            if remaining != last_second:

                print(
                    f"Time Left: "
                    f"{remaining:2d} sec | "
                    f"Packets: "
                    f"{packet_count}"
                )

                last_second = remaining

            # --------------------------------------------------
            # CHECK SERIAL BUFFER
            # --------------------------------------------------

            if ser.in_waiting > 0:

                line = (
                    ser.readline()
                    .decode(errors="ignore")
                    .strip()
                )

                if line.startswith("CSI_DATA"):

                    packet_count += 1

                    writer.writerow([
                        packet_count,
                        line
                    ])

            else:

                # Prevent unnecessary CPU usage
                time.sleep(0.001)

    # ------------------------------------------------------
    # CLOSE SERIAL
    # ------------------------------------------------------

    ser.close()

    # ------------------------------------------------------
    # RECORDING RESULT
    # ------------------------------------------------------

    print(f"\n✓ Saved: {filename}")

    print(
        f"Packets: {packet_count}"
    )

    print(
        f"Average Rate: "
        f"{packet_count / DURATION:.2f} "
        f"packets/sec"
    )

    # ------------------------------------------------------
    # BREAK BETWEEN SAMPLES
    # ------------------------------------------------------

    if sample < num_samples - 1:

        print(
            f"\nNext recording in "
            f"{BREAK_TIME} seconds..."
        )

        time.sleep(BREAK_TIME)


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 50)
print("Dataset Collection Completed Successfully!")
print("=" * 50)

print(
    f"Activity      : {activity}"
)

print(
    f"Samples Saved : {num_samples}"
)

print(
    f"Folder        : {folder}"
)