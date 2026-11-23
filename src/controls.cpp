#include "controls.h"
#include <algorithm>
#include <cstring>

namespace stories {
uint8_t readButtons(int front, int side, bool power) {
  uint8_t raw = power ? buttonBit(Button::Power) : 0;
  if (front >= 0 && front <= 3900)
    raw |= buttonBit(front > 3100 ? Button::Back : front > 2090 ? Button::Confirm
                           : front > 750 ? Button::Left : Button::Right);
  if (side >= 0 && side <= 3900) raw |= buttonBit(side > 1120 ? Button::Up : Button::Down);
  return raw;
}

uint8_t Buttons::sample(uint8_t raw, uint32_t now) {
  uint8_t pressed = 0;
  for (unsigned i = 0; i < 7; ++i) {
    const uint8_t mask = 1u << i;
    if ((raw ^ candidate_) & mask) {
      candidate_ ^= mask;
      changed_[i] = now;
    }
    if (((candidate_ ^ stable_) & mask) && now - changed_[i] >= 25) {
      stable_ ^= mask;
      pressed |= stable_ & mask;
    }
  }
  return pressed;
}

bool Buttons::wantsSleep(uint32_t now) {
  const bool down = powerDown();
  if (!armed_) {
    if (down) releasing_ = false;
    else if (!releasing_) { releasing_ = true; releasedAt_ = now; }
    else if (now - releasedAt_ >= 40) armed_ = true;
    return false;
  }
  if (down) {
    if (!holding_) { holding_ = true; pressedAt_ = now; fired_ = false; }
    if (!fired_ && now - pressedAt_ >= 1200) { fired_ = true; return true; }
  } else if (holding_) {
    holding_ = false;
    return !fired_;
  }
  return false;
}

namespace {
constexpr const char* keys[4][10] = {
  {"q","w","e","r","t","y","u","i","o","p"},
  {"a","s","d","f","g","h","j","k","l","'"},
  {"z","x","c","v","b","n","m",",",".","?"},
  {"1","2","3","4","5","6","7","8","9","0"},
};
constexpr const char* shifted[3][10] = {
  {"Q","W","E","R","T","Y","U","I","O","P"},
  {"A","S","D","F","G","H","J","K","L","\""},
  {"Z","X","C","V","B","N","M",":",";","!"},
};
constexpr const char* actions[] = {"SHIFT", "SPACE", "DEMO", "GO"};
constexpr int counts[] = {10, 10, 10, 10, 4};
}

const char* Keyboard::labels(int row, int col, bool shift) {
  return row == 4 ? actions[col] : shift && row < 3 ? shifted[row][col] : keys[row][col];
}

Action Keyboard::press(Button b) {
  const auto append = [&](char ch) {
    if (length < MaxPrompt) { prompt[length++] = ch; prompt[length] = '\0'; }
  };
  if (b == Button::Back && length) prompt[--length] = '\0';
  if (b == Button::Left) col = (col + counts[row] - 1) % counts[row];
  if (b == Button::Right) col = (col + 1) % counts[row];
  if (b == Button::Up || b == Button::Down) {
    const int next = (row + (b == Button::Down ? 1 : Rows - 1)) % Rows;
    if (next == 4) col = col * 4 / 10;
    if (row == 4) col = col * 10 / 4 + 1;
    row = next;
  }
  if (b != Button::Confirm) return Action::None;
  if (row < 4) append(*labels(row, col, shift));
  else if (col == 0) shift = !shift;
  else if (col == 1) append(' ');
  else if (col == 2) {
    std::strcpy(prompt, "Once upon a time");
    length = std::strlen(prompt);
    col = 3;
  } else return Action::Generate;
  return Action::None;
}
}
