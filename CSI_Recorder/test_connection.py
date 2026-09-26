from serial_manager import connect_serial

try:
    ser = connect_serial()
    print("ESP32 connected successfully!")
    ser.close()
except Exception as e:
    print(f"Error: {e}")