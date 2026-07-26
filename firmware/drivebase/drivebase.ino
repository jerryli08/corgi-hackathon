/*
  Corgi drivebase — 2x Axon Max continuous-rotation servos (differential drive)

  Wiring:
    Left  servo signal -> D9
    Right servo signal -> D10
    Servo power        -> 7.4V battery (NOT Arduino 5V)
    Servo GND          -> battery GND + Arduino GND (common ground required)

  Serial protocol (115200 baud, newline-terminated):
    DRIVE,<left_us>,<right_us>   // 1000-2000, 1500=stop
    STOP
    PING
    WD,<ms>                      // failsafe timeout, 0 disables

  The failsafe is the important part. If the host crashes, unplugs, or simply stops
  talking, the wheels stop on their own after WATCHDOG_MS. The host has its own
  watchdog for the case where a command is late rather than absent; this one covers
  the case where there is no host left to time anything.
*/

#include <Servo.h>

const int PIN_LEFT = 9;
const int PIN_RIGHT = 10;
const int STOP_US = 1500;
const int MIN_US = 1000;
const int MAX_US = 2000;

// Set to true if a wheel runs the wrong way in DRIVE
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = true;  // typical for opposite-mounted wheel

const unsigned long DEFAULT_WATCHDOG_MS = 1500;

Servo leftServo;
Servo rightServo;

unsigned long watchdogMs = DEFAULT_WATCHDOG_MS;
unsigned long lastCommandMs = 0;
bool rolling = false;

int clampUs(int us) {
  if (us < MIN_US) return MIN_US;
  if (us > MAX_US) return MAX_US;
  return us;
}

int maybeInvert(int us, bool invert) {
  if (!invert) return us;
  return STOP_US - (us - STOP_US);
}

void writeWheels(int leftUs, int rightUs) {
  leftUs = clampUs(leftUs);
  rightUs = clampUs(rightUs);
  rolling = (leftUs != STOP_US) || (rightUs != STOP_US);
  leftServo.writeMicroseconds(maybeInvert(leftUs, INVERT_LEFT));
  rightServo.writeMicroseconds(maybeInvert(rightUs, INVERT_RIGHT));
}

void stopWheels() {
  writeWheels(STOP_US, STOP_US);
}

void setup() {
  Serial.begin(115200);
  leftServo.attach(PIN_LEFT);
  rightServo.attach(PIN_RIGHT);
  stopWheels();
  lastCommandMs = millis();
  Serial.println(F("OK drivebase ready"));
}

void handleLine(String line) {
  line.trim();
  line.toUpperCase();
  if (line.length() == 0) return;

  lastCommandMs = millis();

  if (line == "STOP" || line == "S") {
    stopWheels();
    Serial.println(F("OK STOP"));
    return;
  }

  if (line == "PING") {
    Serial.println(F("OK PONG"));
    return;
  }

  if (line.startsWith("WD,")) {
    long ms = line.substring(3).toInt();
    if (ms < 0) {
      Serial.println(F("ERR WD usage: WD,<ms>"));
      return;
    }
    watchdogMs = (unsigned long)ms;
    Serial.print(F("OK WD "));
    Serial.println(watchdogMs);
    return;
  }

  if (line.startsWith("DRIVE,")) {
    int c1 = line.indexOf(',');
    int c2 = line.indexOf(',', c1 + 1);
    if (c1 < 0 || c2 < 0) {
      Serial.println(F("ERR DRIVE usage: DRIVE,<left_us>,<right_us>"));
      return;
    }
    int leftUs = line.substring(c1 + 1, c2).toInt();
    int rightUs = line.substring(c2 + 1).toInt();
    writeWheels(leftUs, rightUs);
    Serial.print(F("OK DRIVE "));
    Serial.print(clampUs(leftUs));
    Serial.print(F(" "));
    Serial.println(clampUs(rightUs));
    return;
  }

  Serial.println(F("ERR unknown command"));
}

void serviceWatchdog() {
  if (!rolling || watchdogMs == 0) return;
  if (millis() - lastCommandMs < watchdogMs) return;
  stopWheels();
  Serial.println(F("WARN watchdog stop"));
}

void loop() {
  static String buf;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (buf.length() > 0) {
        handleLine(buf);
        buf = "";
      }
    } else {
      buf += c;
      if (buf.length() > 64) buf = "";
    }
  }
  serviceWatchdog();
}
