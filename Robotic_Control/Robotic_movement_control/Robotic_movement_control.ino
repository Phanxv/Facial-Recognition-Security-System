#include <Servo.h>

String payload;
Servo MG996R;
int pos = 1500;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
  MG996R.attach(9);
}

void  loop() {
  while (!Serial.available());
  payload = Serial.readString();
  if(payload == "MID"){
    MG996R.writeMicroseconds(pos);
  }
  if(payload == "RIGHT"){
    if(pos >= 0){
      MG996R.writeMicroseconds(pos);
      delay(1);
      pos-=10;
    }
  }
  if(payload == "LEFT"){
    if(pos <= 3000){
      MG996R.writeMicroseconds(pos);
      delay(1);
      pos+=10;
    }
  }
}
