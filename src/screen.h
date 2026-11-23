#pragma once
#include <cstdint>
#include <cstddef>
namespace stories {
class Screen {
 public:
  explicit Screen(uint8_t* fb) : fb_(fb) {}
  void clear();
  void rect(int x, int y, int w, int h, bool ink = true);
  void text(int x, int y, const char* text, int size = 2, bool ink = true);
  void center(int x, int y, const char* text, int size = 2, bool ink = true);
  void paragraph(int x, int y, const char* text, int columns, int lines, int size = 2);
 private:
  uint8_t* fb_;
};
}
