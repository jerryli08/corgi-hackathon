import argparse
import sys
import time

import serial
from serial.tools import list_ports


# Serial settings
DEFAULT_BAUD_RATE = 115200
SERIAL_TIMEOUT_SECONDS = 1.0
ARDUINO_RESET_WAIT_SECONDS = 2.0

# Continuous servo settings. For most continuous servos:
# 90 is stop, below 90 spins one direction, above 90 spins the other.
SERVO_STOP = 90
SERVO_SLOW = 18
SERVO_FAST = 35
COMMAND_DELAY_SECONDS = 1.2


def list_serial_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")


def open_serial(port, baud_rate):
    try:
        connection = serial.Serial(port, baud_rate, timeout=SERIAL_TIMEOUT_SECONDS)
    except serial.SerialException as exc:
        raise RuntimeError(f"Could not open serial port {port}: {exc}") from exc

    time.sleep(ARDUINO_RESET_WAIT_SECONDS)
    connection.reset_input_buffer()
    return connection


def send_servo_command(connection, left, right):
    left = int(max(0, min(180, left)))
    right = int(max(0, min(180, right)))
    message = f"L:{left} R:{right}\n"
    connection.write(message.encode("ascii"))
    connection.flush()
    print(f"sent: {message.strip()}")

    response = connection.readline().decode("ascii", errors="replace").strip()
    if response:
        print(f"arduino: {response}")


def run_sequence(connection):
    tests = [
        ("STOP", SERVO_STOP, SERVO_STOP),
        ("FORWARD SLOW", SERVO_STOP + SERVO_SLOW, SERVO_STOP - SERVO_SLOW),
        ("STOP", SERVO_STOP, SERVO_STOP),
        ("BACKWARD SLOW", SERVO_STOP - SERVO_SLOW, SERVO_STOP + SERVO_SLOW),
        ("STOP", SERVO_STOP, SERVO_STOP),
        ("LEFT TURN", SERVO_STOP - SERVO_SLOW, SERVO_STOP - SERVO_SLOW),
        ("STOP", SERVO_STOP, SERVO_STOP),
        ("RIGHT TURN", SERVO_STOP + SERVO_SLOW, SERVO_STOP + SERVO_SLOW),
        ("STOP", SERVO_STOP, SERVO_STOP),
    ]

    for label, left, right in tests:
        print(f"\n{label}")
        send_servo_command(connection, left, right)
        time.sleep(COMMAND_DELAY_SECONDS)

    send_servo_command(connection, SERVO_STOP, SERVO_STOP)


def interactive_mode(connection):
    print("Interactive mode.")
    print("Commands: stop, f, b, l, r, q")
    print("Raw servo command: two numbers, for example: 108 72")

    while True:
        text = input("> ").strip().lower()
        if text in {"q", "quit", "exit"}:
            send_servo_command(connection, SERVO_STOP, SERVO_STOP)
            break
        if text in {"stop", "s"}:
            send_servo_command(connection, SERVO_STOP, SERVO_STOP)
        elif text in {"f", "forward"}:
            send_servo_command(connection, SERVO_STOP + SERVO_SLOW, SERVO_STOP - SERVO_SLOW)
        elif text in {"b", "back", "backward"}:
            send_servo_command(connection, SERVO_STOP - SERVO_SLOW, SERVO_STOP + SERVO_SLOW)
        elif text in {"l", "left"}:
            send_servo_command(connection, SERVO_STOP - SERVO_SLOW, SERVO_STOP - SERVO_SLOW)
        elif text in {"r", "right"}:
            send_servo_command(connection, SERVO_STOP + SERVO_SLOW, SERVO_STOP + SERVO_SLOW)
        else:
            parts = text.replace(",", " ").split()
            if len(parts) != 2:
                print("Unknown command. Use stop, f, b, l, r, q, or two numbers like: 108 72")
                continue
            try:
                left, right = int(parts[0]), int(parts[1])
            except ValueError:
                print("Servo values must be numbers from 0 to 180.")
                continue
            send_servo_command(connection, left, right)


def parse_args():
    parser = argparse.ArgumentParser(description="Test Python serial control of two Arduino continuous servos.")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit.")
    parser.add_argument("--port", help="Arduino serial port, for example COM3.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE, help="Serial baud rate.")
    parser.add_argument("--interactive", action="store_true", help="Use keyboard interactive mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_serial_ports()
        return 0

    if not args.port:
        print("ERROR: Provide --port COMx, or run --list to find the Arduino port.", file=sys.stderr)
        return 1

    try:
        with open_serial(args.port, args.baud) as connection:
            if args.interactive:
                interactive_mode(connection)
            else:
                run_sequence(connection)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
