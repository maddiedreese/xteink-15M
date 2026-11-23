#pragma once
#include "model.h"

namespace stories {
constexpr size_t App0Address = 0x10000, App1Address = 0x650000;
constexpr size_t LayersOffset = 0x200000 - App0Address;
constexpr size_t TokenizerOffset = 0x540000 - App0Address;
constexpr size_t TokenizerBytes = 529881;
constexpr size_t SlotSize = 0x640000;
bool openFlash(const uint8_t*& layers, Reader& embedding, const uint8_t*& tokenizer);
}
