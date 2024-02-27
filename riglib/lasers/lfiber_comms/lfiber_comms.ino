/*
Communication between usb serial and TTL optical switch

Communication protocol
======================
Input a single byte
0-15: select channel
16: reset
>16: check ready and error status

Output from >16 command
Bit 0: ready
Bit 1: error

TTL operation
=============
Pin I/O       Signal        Description
--- --------- ------------- -----------
1   Input     D0            Bit0-Bit3; D0 is low;
2   Input     D1  
3   Input     D2
4   Input     D3            D3 is high.
5   Input     RESET         TTL, Low level reset to channel 0.
                            High level means channel selection bits are effective.
6   Out       READY         TTL, Ready (High = not ready, Low = ready)
7   Out       ERROR         TTL, Error OR Failure , (High = error, Low = no error)
8   Power     GND           Ground
9   Power     VCC           5.0±5% VDC Power Supply (Max. 500mA)

Written by Leo Scholl, 2024
*/

#define D0 2
#define D1 3
#define D2 4
#define D3 5
#define RST 6
#define RDY 7
#define ERR 8

byte inByte = 0;
bool is_ready = false;
bool waiting_cmd = false;
bool active_cmd = false;
unsigned long const timeout = 1000;
unsigned long elapsed = 0;

void setup() {

  Serial.begin(115200);
  while (!Serial) {
    ;  // wait for serial port to connect. Needed for native USB port only
  }

  pinMode(D0, OUTPUT);
  pinMode(D1, OUTPUT);
  pinMode(D2, OUTPUT);
  pinMode(D3, OUTPUT);
  pinMode(RST, OUTPUT);
  digitalWrite(D0, 0);
  digitalWrite(D1, 0);
  digitalWrite(D2, 0);
  digitalWrite(D3, 0);
  digitalWrite(RST, 0);

  pinMode(RDY, INPUT);
  pinMode(ERR, INPUT);
}

void loop() {

  // check for new commands (without overwriting old ones)
  if (!waiting_cmd && Serial.available() > 0) {

    inByte = Serial.read();
    if (inByte > 16) {

      // send ready status
      Serial.write(is_ready);

    } else {

      // flag to move the optical switch
      waiting_cmd = true;
    }

  }

  // if there is a waiting command, try to move
  if (waiting_cmd && is_ready) {

    if (inByte == 16) {
      
      // reset
      digitalWrite(RST, 0);
      waiting_cmd = false;
      active_cmd = true;
      elapsed = millis();
      
    } else {

      // select channel
      digitalWrite(D0, bitRead(inByte, 0));
      digitalWrite(D1, bitRead(inByte, 1));
      digitalWrite(D2, bitRead(inByte, 2));
      digitalWrite(D3, bitRead(inByte, 3));

      // set channel active
      digitalWrite(RST, 1);
      waiting_cmd = false;
      active_cmd = true;
      elapsed = millis();
      
    }
  }

  // check ready status
  if (!is_ready && !digitalRead(RDY)) { // active low
    is_ready = true;
    
  } else if (is_ready && digitalRead(RDY)) {
    is_ready = false;
  }  

  // if there is an active command, send a message when it's finished
  if (active_cmd && (is_ready || millis() - elapsed > timeout)) {
          
    // send ready status
    Serial.write(is_ready);
    active_cmd = false;   
  }

}
