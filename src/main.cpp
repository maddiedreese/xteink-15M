#include <Arduino.h>
#include <BoardConfig.h>
#include <EInkDisplay.h>
#include <PowerManager.h>
#include <XteinkDetect.h>
#include <driver/gpio.h>
#include <esp_random.h>
#include <esp_sleep.h>
#include <esp_timer.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#include "controls.h"
#include "flash.h"
#include "model.h"
#include "screen.h"

namespace {
using namespace stories;
EInkDisplay display(8, 10, 21, 4, 5, 6);
Model model;
Tokenizer tokenizer;
Buttons buttons;
Keyboard keyboard;
alignas(4) uint8_t scratch[16 * 1024];
char story[4096];
size_t storySize = 0;
uint16_t promptTokens[MaxPrompt + 2];
int promptCount, position, nextToken, generated;
uint32_t rng, refreshes, lastYield;
int64_t started, decodeTime, firstToken;
bool dirty = true, powerOff, stopRequested, bench, serialReady;
enum class Page { Compose, Running, Result, Error };
Page page = Page::Compose;
const char* message = "";
constexpr int Margin = 24;

void powerDown() {
  gpio_hold_dis(GPIO_NUM_12);
  pinMode(12, OUTPUT);
  digitalWrite(12, HIGH);
  gpio_hold_en(GPIO_NUM_12);
  Serial.flush();
  Serial.end();
  gpio_hold_dis(GPIO_NUM_13);
  gpio_set_direction(GPIO_NUM_13, GPIO_MODE_OUTPUT);
  gpio_set_level(GPIO_NUM_13, 0);
  gpio_hold_en(GPIO_NUM_13);
  freeink::PowerManager::powerDownRailsForSleep();
  freeink::PowerManager::deepSleepUntilPowerButton();
  ESP.restart();
}

void fail(const char* text) {
  message = text;
  page = Page::Error;
  dirty = true;
  Serial.printf("ERROR %s\n", text);
}

void append(char ch) {
  if (storySize + 1 >= sizeof(story)) return;
  story[storySize++] = ch;
  story[storySize] = 0;
}

void appendPiece(int token) {
  const char* text = tokenizer.piece(token);
  const int left = std::strlen(text);
  if (left == 6 && std::strncmp(text, "<0x", 3) == 0 && text[5] == '>') {
    char* end = nullptr;
    const auto byte = std::strtoul(text + 3, &end, 16);
    if (*end == '>') {
      if ((byte >= 32 && byte < 127) || byte == '\n') append(char(byte));
      return;
    }
  }
  if (!storySize && *text == ' ') ++text;
  for (; *text; ++text) {
    const uint8_t ch = *text;
    if (ch == '\n' || (ch >= 32 && ch < 127)) append(*text);
    else if ((ch & 0xc0) == 0xc0) append('?');
  }
}

void show() {
  if (!dirty || !display.framebufferReady()) return;
  Screen screen(display.getFrameBuffer());
  screen.clear();
  screen.text(Margin, 30, "TINY STORIES", 4);
  screen.text(Margin, 80, "STORIES15M / XTEINK X4", 2);
  screen.rect(Margin, 111, 432, 3);
  if (page == Page::Compose) {
    screen.text(Margin, 140, "PROMPT", 2);
    screen.paragraph(Margin, 182, keyboard.length ? keyboard.prompt : "Type a beginning, or try DEMO.",
                     35, 8, 1);
    if (*message) screen.paragraph(Margin, 326, message, 51, 4, 1);
    screen.rect(Margin, 382, 432, 1);
    for (int row = 0; row < Keyboard::Rows; ++row) {
      const int count = row == 4 ? 4 : 10;
      const int pitch = row == 4 ? 108 : 43;
      const int y = 409 + row * 57;
      for (int col = 0; col < count; ++col) {
        const int x = Margin + col * pitch;
        const bool selected = row == keyboard.row && col == keyboard.col;
        screen.rect(x, y, pitch - 5, 47);
        if (!selected) screen.rect(x + 2, y + 2, pitch - 9, 43, false);
        screen.center(x + (pitch - 5) / 2, y + 15, Keyboard::labels(row, col, keyboard.shift),
                      2, !selected);
      }
    }
    screen.text(Margin, 724, "ARROWS MOVE  /  OK TYPES", 2);
    screen.text(Margin, 755, "BACK DELETE  /  POWER OFF", 2);
  } else if (page == Page::Error) {
    screen.text(Margin, 150, "NOT READY", 3);
    screen.paragraph(Margin, 207, message, 27, 8);
    screen.text(Margin, 755, "POWER OFF", 2);
  } else {
    screen.paragraph(Margin, 141, storySize ? story : "Starting a story...", 27, 24);
    screen.rect(Margin, 697, 432, 2);
    if (page == Page::Running) {
      char label[40];
      if (position < promptCount - 1)
        std::snprintf(label, sizeof(label), "READING PROMPT %d/%d", position, promptCount);
      else std::snprintf(label, sizeof(label), "WRITING / %d TOKENS", generated);
      screen.text(Margin, 717, label);
      screen.text(Margin, 755, "BACK STOP  /  POWER OFF", 2);
    } else {
      screen.paragraph(Margin, 718, message, 53, 1, 1);
      screen.text(Margin, 755, "BACK EDIT  /  OK AGAIN", 2);
    }
  }
  // Blocking refresh reseeds the SSD1677's previous-frame RAM without a second framebuffer.
  display.displayBuffer(refreshes % 30 == 0 ? EInkDisplay::HALF_REFRESH : EInkDisplay::FAST_REFRESH);
  ++refreshes;
  dirty = false;
}

void sleepNow() {
  if (display.framebufferReady()) {
    Screen screen(display.getFrameBuffer());
    screen.clear();
    screen.center(240, 343, "TINY STORIES", 3);
    screen.center(240, 409, "POWER TO WAKE", 2);
    display.displayBuffer(EInkDisplay::FULL_REFRESH, true);
    display.deepSleep();
  }
  powerDown();
}

void start(bool benchmark = false) {
  const char* prompt = benchmark ? "Once upon a time" : keyboard.prompt;
  promptCount = tokenizer.encode(prompt, promptTokens, sizeof(promptTokens) / sizeof(promptTokens[0]));
  if (promptCount < 1 || promptCount >= model.context()) {
    message = "Prompt too long. Delete a few characters.";
    page = Page::Compose;
    dirty = true;
    return;
  }
  bench = benchmark;
  rng = bench ? 42 : (esp_random() | 1);
  storySize = std::strlen(prompt);
  std::memcpy(story, prompt, storySize + 1);
  position = generated = 0;
  nextToken = Bos;
  started = esp_timer_get_time();
  decodeTime = firstToken = 0;
  stopRequested = false;
  page = Page::Running;
  dirty = true;
  Serial.printf("%s START context=%d prompt_tokens=%d\n", bench ? "BENCH" : "RUN",
                model.context(), promptCount);
}

void finish(const char* why) {
  message = why;
  page = Page::Result;
  dirty = true;
  Serial.printf("%s DONE reason=%s tokens=%d decode_us=%" PRId64 " first_us=%" PRId64
                " total_us=%" PRId64 "\n", bench ? "BENCH" : "RUN", why, generated,
                decodeTime, firstToken, esp_timer_get_time() - started);
  Serial.printf("STORY %s\n", story);
}

void input() {
  const uint32_t now = millis();
  const auto mask = buttons.sample(readButtons(analogRead(1), analogRead(2), digitalRead(3) == LOW), now);
  if (buttons.wantsSleep(now)) powerOff = true;
  if (page == Page::Running) {
    if (mask & buttonBit(Button::Back)) stopRequested = true;
    return;
  }
  if (page == Page::Result) {
    if (mask & buttonBit(Button::Back)) { page = Page::Compose; message = ""; dirty = true; }
    else if (mask & buttonBit(Button::Confirm)) start(bench);
    return;
  }
  if (page != Page::Compose) return;
  for (unsigned i = 0; i < unsigned(Button::Power); ++i) {
    if (!(mask & (1u << i))) continue;
    message = "";
    dirty = true;
    if (keyboard.press(static_cast<Button>(i)) == Action::Generate) { start(); break; }
  }
}

bool pollWork() {
  const uint32_t now = millis();
  if (now - lastYield >= 16) {
    input();
    delay(1);
    lastYield = now;
  }
  return !powerOff && !stopRequested;
}

void runOne() {
  if (stopRequested) { finish("Stopped"); return; }
  const bool sampling = position >= promptCount - 1;
  const int token = position < promptCount ? promptTokens[position] : nextToken;
  const auto t0 = esp_timer_get_time();
  const int result = model.step(token, position, sampling, rng, {pollWork});
  const auto elapsed = esp_timer_get_time() - t0;
  if (result == -1) {
    if (model.ioFailed()) fail("Could not read model from flash.");
    else finish("Stopped");
    return;
  }
  ++position;
  if (!sampling) return;
  if (result == Eos) { finish("Story complete"); return; }
  nextToken = result;
  appendPiece(result);
  if (!generated) firstToken = esp_timer_get_time() - started;
  decodeTime += elapsed;
  ++generated;
  dirty = true;
  Serial.printf("TOKEN %d id=%d compute_us=%" PRId64 "\n", generated, result, elapsed);
  if (position == model.context()) finish("Context limit reached");
  else if (bench && generated >= 16) finish("Benchmark complete");
}

void printStatus() {
  Serial.printf("READY controller=%s context=%d free_heap=%u\n", display.getControllerName(),
                model.context(), ESP.getFreeHeap());
}
}  // namespace

void setup() {
  BoardConfig::holdPowerRails();
  gpio_deep_sleep_hold_dis();
  pinMode(3, INPUT_PULLUP);
  pinMode(20, INPUT);
  Serial.begin(115200);
  const auto reset = esp_reset_reason();
  const bool cold = reset == ESP_RST_POWERON || reset == ESP_RST_BROWNOUT || reset == ESP_RST_DEEPSLEEP;
  const bool initiallyHeld = digitalRead(3) == LOW;
  if (cold && !digitalRead(20)) {
    delay(10);
    if (!initiallyHeld || digitalRead(3) != LOW) powerDown();
  }
  pinMode(1, INPUT);
  pinMode(2, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(1, ADC_11db);
  analogSetPinAttenuation(2, ADC_11db);
  gpio_hold_dis(GPIO_NUM_12);
  pinMode(12, OUTPUT);
  digitalWrite(12, HIGH);
  freeink::applyXteinkDisplayController();
  display.begin();
  display.setSingleBufferFastDiff(true);
  const uint8_t *layers = nullptr, *vocab = nullptr;
  Reader embedding;
  if (!display.framebufferReady()) fail("Could not allocate display buffer.");
  else if (!openFlash(layers, embedding, vocab) || !tokenizer.begin(vocab, TokenizerBytes))
    fail("Model missing. Flash the model and tokenizer.");
  else {
    for (int ctx = MaxContext; ctx >= MinContext; ctx -= 8) {
      const unsigned kv = Layers * ctx * (2 * Dim + 2 * Heads * sizeof(float));
      if (ESP.getFreeHeap() > kv + 24 * 1024 &&
          model.begin(layers, embedding, ctx, scratch, sizeof(scratch))) break;
    }
    if (!model.context()) fail("Model or context initialization failed.");
    Serial.printf("EMBEDDING mode=%s\n", embedding.mapped ? "mapped" : "read");
  }
}

void loop() {
  input();
  if (powerOff) { sleepNow(); return; }
  if (Serial && !serialReady) { serialReady = true; printStatus(); }
  while (Serial.available()) {
    const int command = Serial.read();
    if (command == '?') printStatus();
    if (command == 'b' && model.context() && page != Page::Running) start(true);
  }
  show();
  if (page == Page::Running) runOne();
  else delay(5);
}
