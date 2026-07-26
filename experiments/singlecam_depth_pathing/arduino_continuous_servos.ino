#include <Servo.h>

const int LEFT_SERVO_PIN = 9;
const int RIGHT_SERVO_PIN = 10;
const int AXIS_SERVO_PIN = 3;

const int SERVO_STOP = 90;
const unsigned long COMMAND_TIMEOUT_MS = 1000;

Servo leftServo;
Servo rightServo;
Servo axisServo;
unsigned long lastCommandMs = 0;

void setup() {
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);
  axisServo.attach(AXIS_SERVO_PIN);
  stopServos();

  Serial.begin(115200);
  Serial.setTimeout(50);
  lastCommandMs = millis();
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      handleCommand(line);
    }
  }

  if (millis() - lastCommandMs > COMMAND_TIMEOUT_MS) {
    stopServos();
  }
}

void handleCommand(String line) {
  int leftValue = SERVO_STOP;
  int rightValue = SERVO_STOP;
  int axisValue = SERVO_STOP;

  int leftIndex = line.indexOf("L:");
  int rightIndex = line.indexOf("R:");
  int axisIndex = line.indexOf("A:");

  if (leftIndex == -1 || rightIndex == -1) {
    Serial.println("ERR expected L:<0-180> R:<0-180> optional A:<0-180>");
    return;
  }

  leftValue = line.substring(leftIndex + 2).toInt();
  rightValue = line.substring(rightIndex + 2).toInt();
  if (axisIndex != -1) {
    axisValue = line.substring(axisIndex + 2).toInt();
  }

  leftValue = constrain(leftValue, 0, 180);
  rightValue = constrain(rightValue, 0, 180);
  axisValue = constrain(axisValue, 0, 180);

  leftServo.write(leftValue);
  rightServo.write(rightValue);
  axisServo.write(axisValue);
  lastCommandMs = millis();

  Serial.print("OK L:");
  Serial.print(leftValue);
  Serial.print(" R:");
  Serial.print(rightValue);
  Serial.print(" A:");
  Serial.println(axisValue);
}

void stopServos() {
  leftServo.write(SERVO_STOP);
  rightServo.write(SERVO_STOP);
  axisServo.write(SERVO_STOP);
}
