#include "controls.h"
#include "model.h"
#include "screen.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace stories;
static std::vector<uint8_t> embedding;
std::vector<uint8_t> load(const char* path) {
  std::ifstream input(path, std::ios::binary);
  assert(input);
  return {std::istreambuf_iterator<char>(input), {}};
}
bool readEmbedding(size_t at, void* data, size_t size) {
  if (at + size > embedding.size()) return false;
  std::memcpy(data, embedding.data() + at, size);
  return true;
}

void keyboardTest() {
  Keyboard keys;
  auto press = [&](Button b) { return keys.press(b); };
  press(Button::Confirm);                     // a
  press(Button::Right);
  press(Button::Confirm);                     // s
  assert(std::strcmp(keys.prompt, "as") == 0);
  press(Button::Back);
  assert(std::strcmp(keys.prompt, "a") == 0);
  press(Button::Down);
  press(Button::Down);
  press(Button::Down);
  assert(keys.row == 4 && keys.col == 0);
  press(Button::Confirm);                     // shift
  assert(keys.shift && std::strcmp(Keyboard::labels(1, 0, keys.shift), "A") == 0);
  press(Button::Right);
  press(Button::Confirm);                     // space
  assert(std::strcmp(keys.prompt, "a ") == 0);
  press(Button::Right);
  press(Button::Confirm);                     // demo
  assert(keys.col == 3 && std::strcmp(keys.prompt, "Once upon a time") == 0);
  assert(press(Button::Confirm) == Action::Generate);
  keys.row = 0;
  keys.col = 0;
  for (size_t i = 0; i < MaxPrompt; ++i) press(Button::Confirm);
  assert(keys.length == MaxPrompt && keys.prompt[MaxPrompt] == '\0');

  assert(readButtons(3600, 2000, false) == (buttonBit(Button::Back) | buttonBit(Button::Up)));
  assert(readButtons(2600, 4095, false) == buttonBit(Button::Confirm));
  assert(readButtons(1500, 4095, false) == buttonBit(Button::Left));
  assert(readButtons(100, 300, true) ==
         (buttonBit(Button::Right) | buttonBit(Button::Down) | buttonBit(Button::Power)));
  Buttons buttons;
  auto raw = buttonBit(Button::Power);
  auto tick = [&](uint32_t now, uint8_t mask) {
    buttons.sample(mask, now);
    return buttons.wantsSleep(now);
  };
  assert(!tick(0, raw) && !tick(30, raw));      // held at boot
  assert(!tick(60, 0) && !tick(90, 0) && !tick(140, 0));
  assert(!tick(200, raw) && !tick(230, raw));
  assert(!tick(300, 0) && tick(330, 0));       // short press
  assert(!tick(400, raw) && !tick(430, raw));
  assert(tick(1630, raw) && !tick(1640, raw)); // hold fires only once

  std::array<uint8_t, 48000> frame{};
  Screen screen(frame.data());
  screen.clear();
  screen.rect(0, 0, 1, 1);
  assert(frame[47900] == 0x7f && frame[0] == 0xff);
  screen.rect(479, 799, 1, 1);
  assert(frame[99] == 0xfe);
  screen.paragraph(24, 100, "a test paragraph\nwith a line break", 8, 10);
}

std::string generate(const uint8_t* layerData, Reader reader, const Tokenizer& t) {
  Model model;
  alignas(4) uint8_t scratch[4096];
  assert(model.begin(layerData, reader, MinContext, scratch, sizeof(scratch)));
  uint16_t prompt[32];
  int count = t.encode("Once upon a time", prompt, 32);
  assert(count == 5 && prompt[0] == Bos && prompt[1] == 9038 &&
         prompt[2] == 2501 && prompt[3] == 263 && prompt[4] == 931);
  assert(t.encode("x", prompt, 1) < 0);
  assert(t.encode("Once upon a time", prompt, 32) == count);
  uint32_t random = 0;
  int token = Bos;
  std::string story = "Once upon a time";
  for (int pos = 0; pos < 24; ++pos) {
    const bool sample = pos >= count - 1;
    float logit = 0;
    token = model.step(pos < count ? prompt[pos] : token, pos, sample, random, {}, &logit);
    if (!sample) { assert(token == 0); continue; }
    assert(token >= Eos && std::isfinite(logit));
    if (token == Eos) break;
    story += t.piece(token);
  }
  assert(model.step(Bos, model.context(), false, random) < 0);
  assert(model.step(Bos, 0, false, random, {[] { return false; }}) < 0);
  return story;
}

int main() {
  keyboardTest();
  embedding = load("assets/embedding.bin");
  const auto layers = load("assets/layers.bin");
  const auto vocab = load("assets/tokenizer.bin");
  assert(embedding.size() == EmbeddingBytes && layers.size() == LayerBytes);
  Tokenizer tokenizer;
  assert(!tokenizer.begin(vocab.data(), 10));
  assert(tokenizer.begin(vocab.data(), vocab.size()));
  const auto story = generate(layers.data(), {embedding.data(), nullptr}, tokenizer);
  assert(story == generate(layers.data(), {nullptr, readEmbedding}, tokenizer));
  std::puts(story.c_str());
  std::fflush(stdout);
  assert(story.find("Once upon a time, there was a little") == 0);
}
