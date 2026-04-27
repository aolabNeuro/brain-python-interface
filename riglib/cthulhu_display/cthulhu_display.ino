/*
  Cthulhu Tongue Display serial grid demo

  Protocol (ASCII lines, newline terminated):
  - G <w> <h> <p0> ... <pN-1>
      Load an 8-bit image frame (0..255), row-major.
  - S <alpha> <threshold>
      Update smoothing alpha (0..1) and activation threshold (0..1).
  - Z
      Clear the display.
  - ?
      Print status.

  Notes:
  - The shield is driven as a 4x4 electrode array (16 channels).
  - Incoming frames can be up to 16x16 and are spatially interpolated via
    bilinear sampling.
  - A temporal 4-phase subpixel scan is used to increase perceived detail.
*/

#include <Cthulhu.h>

const int BAUD_RATE = 115200;
const int MAX_GRID_SIDE = 16;
const int MAX_PIXELS = MAX_GRID_SIDE * MAX_GRID_SIDE;
const int ELECTRODE_SIDE = 4;
const int ELECTRODE_COUNT = ELECTRODE_SIDE * ELECTRODE_SIDE;
const int STIM_CHANNELS = 18;
const int SERIAL_LINE_MAX = 1400;
const unsigned long inputTimeoutMs = 250;

char serialLine[SERIAL_LINE_MAX];
int serialLineLen = 0;

int frameW = 8;
int frameH = 8;
uint8_t inputGrid[MAX_PIXELS];
float smoothedGrid[MAX_PIXELS];

int array[STIM_CHANNELS];
int PP[STIM_CHANNELS];
int Pp[STIM_CHANNELS];
int IN[STIM_CHANNELS];
int IP[STIM_CHANNELS];
int ON[STIM_CHANNELS];

float smoothAlpha = 0.35f;
float activationThreshold = 0.04f;
float intensityGain = 0.45f;
uint8_t scanPhase = 0;
unsigned long lastStimMs = 0;
const unsigned long stimPeriodMs = 10;
unsigned long lastGridInputMs = 0;
bool streamActive = false;

Cthulhu mycthulhu;

float phaseOffsets[4][2] = {
  {0.25f, 0.25f},
  {0.75f, 0.25f},
  {0.25f, 0.75f},
  {0.75f, 0.75f}
};

int clampInt(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

float clampFloat(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

int idx(int row, int col, int width) {
  return row * width + col;
}

void clearFrame() {
  for (int i = 0; i < MAX_PIXELS; i++) {
    inputGrid[i] = 0;
    smoothedGrid[i] = 0.0f;
  }
}

void initStimParams() {
  for (int i = 0; i < STIM_CHANNELS; i++) {
    array[i] = 0;
    PP[i] = 50;
    Pp[i] = 6;
    IN[i] = 3;
    IP[i] = 150;
    ON[i] = 5;
  }
}

void disableStimuli() {
  for (int i = 0; i < STIM_CHANNELS; i++) {
    array[i] = 0;
    Pp[i] = 4;
  }
  mycthulhu.UpdateStimuli(array, PP, Pp, IN, IP, ON);
}

float sampleBilinear(float y, float x) {
  if (frameW < 1 || frameH < 1) return 0.0f;

  x = clampFloat(x, 0.0f, (float)(frameW - 1));
  y = clampFloat(y, 0.0f, (float)(frameH - 1));

  int x0 = (int)x;
  int y0 = (int)y;
  int x1 = clampInt(x0 + 1, 0, frameW - 1);
  int y1 = clampInt(y0 + 1, 0, frameH - 1);

  float fx = x - x0;
  float fy = y - y0;

  float v00 = smoothedGrid[idx(y0, x0, frameW)];
  float v01 = smoothedGrid[idx(y0, x1, frameW)];
  float v10 = smoothedGrid[idx(y1, x0, frameW)];
  float v11 = smoothedGrid[idx(y1, x1, frameW)];

  float top = v00 + fx * (v01 - v00);
  float bottom = v10 + fx * (v11 - v10);
  return top + fy * (bottom - top);
}

void smoothInput() {
  int n = frameW * frameH;
  float beta = 1.0f - smoothAlpha;
  for (int i = 0; i < n; i++) {
    float target = ((float)inputGrid[i]) / 255.0f;
    smoothedGrid[i] = beta * smoothedGrid[i] + smoothAlpha * target;
  }
}

void updateStimFromGrid() {
  smoothInput();

  float phaseX = phaseOffsets[scanPhase][0];
  float phaseY = phaseOffsets[scanPhase][1];
  scanPhase = (scanPhase + 1) % 4;

  array[0] = 0;
  array[1] = 0;
  Pp[0] = 8;
  Pp[1] = 8;

  for (int er = 0; er < ELECTRODE_SIDE; er++) {
    for (int ec = 0; ec < ELECTRODE_SIDE; ec++) {
      float normX = (ec + phaseX) / (float)ELECTRODE_SIDE;
      float normY = (er + phaseY) / (float)ELECTRODE_SIDE;

      float gx = normX * (frameW - 1);
      float gy = normY * (frameH - 1);
      float intensity = sampleBilinear(gy, gx);

      int channel = 2 + er * ELECTRODE_SIDE + ec;
      if (intensity >= activationThreshold) {
        array[channel] = 1;
        int pulse = (int)((4 + intensity * 28) * intensityGain);
        Pp[channel] = clampInt(pulse, 2, 48);
      } else {
        array[channel] = 0;
        Pp[channel] = 2;
      }
    }
  }

  mycthulhu.UpdateStimuli(array, PP, Pp, IN, IP, ON);
}

bool parseGridCommand(char* token) {
  token = strtok(NULL, " ,\t");
  if (token == NULL) return false;
  int w = atoi(token);

  token = strtok(NULL, " ,\t");
  if (token == NULL) return false;
  int h = atoi(token);

  if (w < 1 || h < 1 || w > MAX_GRID_SIDE || h > MAX_GRID_SIDE) {
    return false;
  }

  int n = w * h;
  for (int i = 0; i < n; i++) {
    token = strtok(NULL, " ,\t");
    if (token == NULL) return false;
    int value = clampInt(atoi(token), 0, 255);
    inputGrid[i] = (uint8_t)value;
  }

  frameW = w;
  frameH = h;
  return true;
}

void handleLine(char* line) {
  char* token = strtok(line, " ,\t");
  if (token == NULL) return;

  if (token[0] == 'G') {
    bool ok = parseGridCommand(token);
    if (ok) {
      lastGridInputMs = millis();
      streamActive = true;
      Serial.println("OK G");
    } else {
      Serial.println("ERR G");
    }
  } else if (token[0] == 'S') {
    char* alphaToken = strtok(NULL, " ,\t");
    char* thrToken = strtok(NULL, " ,\t");
    if (alphaToken == NULL || thrToken == NULL) {
      Serial.println("ERR S");
      return;
    }
    smoothAlpha = clampFloat(atof(alphaToken), 0.01f, 1.0f);
    activationThreshold = clampFloat(atof(thrToken), 0.0f, 1.0f);
    Serial.println("OK S");
  } else if (token[0] == 'I') {
    char* gainToken = strtok(NULL, " ,\t");
    if (gainToken == NULL) {
      Serial.println("ERR I");
      return;
    }
    intensityGain = clampFloat(atof(gainToken), 0.0f, 1.0f);
    Serial.println("OK I");
  } else if (token[0] == 'Z') {
    clearFrame();
    streamActive = false;
    disableStimuli();
    Serial.println("OK Z");
  } else if (token[0] == '?') {
    Serial.print("CTHULHU_DISPLAY_TDU ");
    Serial.print(frameW);
    Serial.print("x");
    Serial.print(frameH);
    Serial.print(" alpha=");
    Serial.print(smoothAlpha, 3);
    Serial.print(" thr=");
    Serial.print(activationThreshold, 3);
    Serial.print(" gain=");
    Serial.println(intensityGain, 3);
  } else {
    Serial.println("ERR CMD");
  }
}

void readSerialLines() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      serialLine[serialLineLen] = '\0';
      if (serialLineLen > 0) {
        handleLine(serialLine);
      }
      serialLineLen = 0;
      continue;
    }

    if (serialLineLen < SERIAL_LINE_MAX - 1) {
      serialLine[serialLineLen++] = c;
    } else {
      serialLineLen = 0;
      Serial.println("ERR LINE");
    }
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  mycthulhu.Begin();
  initStimParams();
  clearFrame();
  disableStimuli();
  Serial.println("CTHULHU_DISPLAY_TDU_READY");
}

void loop() {
  readSerialLines();

  unsigned long now = millis();
  if (streamActive && (now - lastGridInputMs) > inputTimeoutMs) {
    streamActive = false;
    clearFrame();
    disableStimuli();
  }

  if (streamActive && (now - lastStimMs) >= stimPeriodMs) {
    lastStimMs = now;
    updateStimFromGrid();
    mycthulhu.Stimulate();
    mycthulhu.Stimulate();
  }
}
