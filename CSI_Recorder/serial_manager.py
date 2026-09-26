import serial
import serial.tools.list_ports
from config import BAUD_RATE


def find_esp32_port():
    """
    Automatically detect the ESP32 serial port.
    """
    ports = serial.tools.list_ports.comports()

    for port in ports:
        description = port.description.lower()

        if (
            "usb" in description
            or "uart" in description
            or "cp210" in description
            or "ch340" in description
            or "silicon labs" in description
        ):
            return port.device

    return None


def connect_serial():
    """
    Connect to the detected ESP32.
    """
    port = find_esp32_port()

    if port is None:
        raise Exception("ESP32 not found. Please connect the board.")

    print(f"Connecting to {port}...")

    ser = serial.Serial(
        port=port,
        baudrate=BAUD_RATE,
        timeout=1
    )

    print("Connection successful.")

    return ser