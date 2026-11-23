#pragma once
#include <cstddef>
#include <cstdint>

namespace stories {
enum class Button : uint8_t { Back, Confirm, Left, Right, Up, Down, Power };
constexpr uint8_t buttonBit(Button b) { return 1u << unsigned(b); }
uint8_t readButtons(int front, int side, bool power);

class Buttons {
 public:
  uint8_t sample(uint8_t raw, uint32_t now);
  bool powerDown() const { return stable_ & buttonBit(Button::Power); }
  bool wantsSleep(uint32_t now);
 private:
  uint8_t candidate_ = 0, stable_ = 0;
  uint32_t changed_[7]{};
  bool armed_ = false, releasing_ = false, holding_ = false, fired_ = false;
  uint32_t releasedAt_ = 0, pressedAt_ = 0;
};

constexpr size_t MaxPrompt = 180;
enum class Action { None, Generate };
struct Keyboard {
  enum { Rows = 5 };
  static const char* labels(int row, int col, bool shift);
  int row = 1, col = 0;
  bool shift = false;
  char prompt[MaxPrompt + 1]{};
  size_t length = 0;
  Action press(Button b);
};
}
