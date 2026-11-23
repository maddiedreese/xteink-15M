#include "screen.h"
#include "font8x8.h"
#include <algorithm>
#include <cstring>

namespace stories {
void Screen::clear() { std::memset(fb_, 0xff, 48000); }

void Screen::rect(int x, int y, int w, int h, bool ink) {
  for (int col = std::max(0, x); col < std::min(480, x + w); ++col) {
    auto* row = fb_ + (479 - col) * 100;
    for (int yy = std::max(0, y); yy < std::min(800, y + h); ++yy) {
      const uint8_t mask = 0x80 >> (yy & 7);
      if (ink) row[yy / 8] &= ~mask;
      else row[yy / 8] |= mask;
    }
  }
}

void Screen::text(int x, int y, const char* text, int size, bool ink) {
  for (auto* p = text; *p; ++p, x += 8 * size) {
    const uint8_t ch = uint8_t(*p);
    const auto* rows = font8x8_basic[ch >= 32 && ch < 127 ? ch : '?'];
    for (int j = 0; j < 8; ++j)
      for (int i = 0; i < 8; ++i)
        if (rows[j] & (1 << i)) rect(x + i * size, y + j * size, size, size, ink);
  }
}

void Screen::center(int x, int y, const char* s, int size, bool ink) {
  text(x - int(std::strlen(s)) * 4 * size, y, s, size, ink);
}

void Screen::paragraph(int x, int y, const char* s, int columns, int lines, int size) {
  if (columns < 1 || columns >= 60) return;
  for (int i = 0; i < lines && *s; ++i) {
    char line[60];
    int n = 0, space = -1;
    while (s[n] && s[n] != '\n' && n < columns) {
      if (s[n] == ' ') space = n;
      ++n;
    }
    int advance = n;
    if (n == columns && s[n] && s[n] != '\n' && s[n] != ' ' && space > 0) {
      n = space;
      advance = space + 1;
    } else if (s[n] == '\n' || s[n] == ' ') ++advance;
    std::memcpy(line, s, n);
    line[n] = 0;
    text(x, y + i * (size * 8 + 6), line, size);
    s += advance;
  }
}
}
