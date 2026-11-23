#pragma once

#include <cstddef>
#include <cstdint>

namespace stories {
constexpr int Dim = 288, Hidden = 768, Layers = 6, Heads = 6, Head = Dim / Heads;
constexpr int Vocab = 32000, Group = 32, MaxContext = 64, MinContext = 32;
constexpr uint16_t Bos = 1, Eos = 2;
constexpr size_t Header = 8;
constexpr size_t stride(int width) { return (width / Group * 18 + 3) & ~size_t(3); }
constexpr size_t EmbeddingBytes = Header + Vocab * stride(Dim);
constexpr size_t LayerBytes = Header + (2 * Layers + 1) * Dim * sizeof(float) +
                              Layers * ((4 * Dim + 2 * Hidden) * stride(Dim) + Dim * stride(Hidden));

struct Reader {
  const uint8_t* mapped = nullptr;
  bool (*read)(size_t offset, void* data, size_t count) = nullptr;
};

struct Work {
  bool (*poll)() = nullptr;
  bool keepGoing() const { return !poll || poll(); }
};

class Model {
 public:
  Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model();
  bool begin(const uint8_t* layers, Reader embedding, int context, uint8_t* scratch,
             size_t scratchBytes);
  int context() const { return context_; }
  // >0: sampled token; 0: prefill; -1: interrupted or failed.
  int step(int token, int position, bool sample, uint32_t& random, Work work = {},
           float* topLogit = nullptr);
  bool ioFailed() const { return ioFailed_; }

 private:
  struct Kv {
    int8_t key[Dim], value[Dim];
    float keyScale[Heads], valueScale[Heads];
  };
  void attention(int layer, int pos);
  bool read(size_t offset, size_t count, const uint8_t*& result);
  int classify(uint32_t& random, Work work, float* topLogit);
  Kv* cache_[Layers]{};
  const float* norms_ = nullptr;
  const uint8_t* matrices_[7]{};
  Reader embedding_{};
  uint8_t* scratch_ = nullptr;
  size_t scratchBytes_ = 0;
  int context_ = 0;
  bool ioFailed_ = false;
  float x_[Dim], xb_[Dim], q_[Dim], k_[Dim], v_[Dim], tmp_[Dim];
  float h_[Hidden], gate_[Hidden], att_[MaxContext];
  float cos_[Head / 2], sin_[Head / 2], freq_[Head / 2];
  int8_t act_[Hidden];
  float scale_[Hidden / Group];
  int bias_[Hidden / Group];
};

class Tokenizer {
 public:
  bool begin(const uint8_t* data, size_t bytes);
  int encode(const char* prompt, uint16_t* tokens, int capacity) const;
  const char* piece(int token) const;
 private:
  int lookup(const char* text) const;
  const uint8_t* data_ = nullptr;
  const char* blob_ = nullptr;
};
}  // namespace stories
