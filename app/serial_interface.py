"""
Optional hardware-in-the-loop interface: reads a single float per line
from a serial device (e.g. an Arduino streaming a sensor reading) in
place of the simulated plant.

If pyserial is not installed or no device is connected, callers should
treat every function here as returning None and fall back to simulation.
"""

import time

try:
    import serial
except ImportError:
    serial = None


DEFAULT_PORT = "COM3"       # Windows default; use "/dev/ttyUSB0" on Linux,
                             # "/dev/tty.usbserial-*" on macOS.
DEFAULT_BAUDRATE = 9600
RESET_DELAY_S = 2.0         # Time to wait after opening the port for
                             # boards (e.g. Arduino) that reset on connect.


def try_serial_connection(port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE, timeout=1):
    """
    Attempt to open a serial connection. Returns the open `Serial` object,
    or None if pyserial isn't installed or the connection fails.
    """
    if serial is None:
        print("[Warning] pyserial is not installed. Run: pip install pyserial")
        return None
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(RESET_DELAY_S)
        return ser
    except Exception as exc:
        print(f"[Warning] Could not connect to serial port {port}: {exc}")
        return None


def get_sensor_data(ser):
    """
    Read and parse one line from the serial device as a float.

    Returns None if the connection is missing, the line is empty, or it
    can't be parsed as a float — callers should skip that simulation step
    rather than crash on a bad reading.
    """
    if ser is None:
        return None
    try:
        line = ser.readline().decode().strip()
        if not line:
            return None
        return float(line)
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def close_connection(ser):
    """Safely close a serial connection, ignoring errors if already closed."""
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
