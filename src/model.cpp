#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace stories {
namespace {
template<class T> T load(const uint8_t* p) {
  T result;
  std::memcpy(&result, p, sizeof(T));
  return result;
}

float weightScale(const uint8_t* p) {
  const uint32_t bits = uint32_t(load<uint16_t>(p + 16)) << 16;
  float scale;
  std::memcpy(&scale, &bits, sizeof(scale));
  return scale;
}

void quantize(const float* x, int n, int8_t* out, float& scale) {
  float norm = 0;
  for (int i = 0; i < n; ++i) norm = std::max(norm, std::fabs(x[i]));
  scale = norm / 127.0f;
  const float inverse = norm > 0 ? 127.0f / norm : 0;
  for (int i = 0; i < n; ++i) {
    const float v = inverse * x[i];
    out[i] = int(v + (v < 0 ? -0.5f : 0.5f));
  }
}

void quantizeGroups(const float* x, int n, int8_t* out, float* scales, int* bias) {
  for (int i = 0; i < n; i += Group) {
    quantize(x + i, Group, out + i, scales[i / Group]);
    int sum = 0;
    for (int j = 0; j < Group; ++j) sum += out[i + j];
    bias[i / Group] = 8 * sum;
  }
}

float dot(const uint8_t* w, int groups, const int8_t* x, const float* scales, const int* bias) {
  float result = 0;
  for (int g = 0; g < groups; ++g, w += 18, x += Group) {
    int sum = 0;
    for (int j = 0; j < 16; ++j) {
      const int a = x[2 * j], b = x[2 * j + 1], p = w[j];
      sum += (p & 15) * a + (p >> 4) * b;
    }
    result += (sum - bias[g]) * (weightScale(w) * scales[g]);
  }
  return result;
}

void matvec(float* out, const uint8_t* w, int width, int rows,
            const int8_t* x, const float* scales, const int* bias) {
  const auto rowBytes = stride(width);
  for (int i = 0; i < rows; ++i, w += rowBytes)
    out[i] = dot(w, width / Group, x, scales, bias);
}

void rms(float* out, const float* x, const float* w) {
  float sum = 0;
  for (int i = 0; i < Dim; ++i) sum += x[i] * x[i];
  const float s = 1.0f / std::sqrt(sum / float(Dim) + 1.e-5f);
  for (int i = 0; i < Dim; ++i) out[i] = x[i] * s * w[i];
}

void softmax(float* x, int n) {
  const float largest = *std::max_element(x, x + n);
  float sum = 0;
  for (int i = 0; i < n; ++i) sum += (x[i] = std::exp(x[i] - largest));
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

constexpr size_t TokIndex = 16;
constexpr size_t TokOrder = TokIndex + Vocab * 8;
constexpr size_t TokBlob = TokOrder + Vocab * 2;
}  // namespace

Model::~Model() {
  for (auto* layer : cache_) std::free(layer);
}

bool Model::begin(const uint8_t* layers, Reader embedding, int context,
                  uint8_t* scratch, size_t scratchBytes) {
  if (context_ || !layers || std::memcmp(layers, "X4LYR001", Header) != 0 ||
      context < MinContext || context > MaxContext ||
      (!embedding.mapped && (!embedding.read || scratchBytes < stride(Dim) || !scratch)))
    return false;
  embedding_ = embedding;
  scratch_ = scratch;
  scratchBytes_ = scratchBytes;
  const uint8_t* magic;
  if (!read(0, Header, magic) || std::memcmp(magic, "X4EMB001", Header) != 0) return false;

  for (auto& layer : cache_) {
    layer = static_cast<Kv*>(std::malloc(context * sizeof(Kv)));
    if (!layer) {
      for (auto& item : cache_) { std::free(item); item = nullptr; }
      return false;
    }
  }

  norms_ = reinterpret_cast<const float*>(layers + Header);
  auto* data = layers + Header + (2 * Layers + 1) * Dim * sizeof(float);
  for (int i = 0; i < 7; ++i) {
    matrices_[i] = data;
    const int rows = (i == 4 || i == 6) ? Hidden : Dim;
    data += Layers * rows * stride(i == 5 ? Hidden : Dim);
  }
  for (int i = 0; i < Head / 2; ++i)
    freq_[i] = std::pow(10000.0f, -2.0f * i / Head);
  context_ = context;
  return true;
}

bool Model::read(size_t offset, size_t count, const uint8_t*& result) {
  if (embedding_.mapped) {
    result = embedding_.mapped + offset;
    return true;
  }
  result = scratch_;
  const bool ok = count <= scratchBytes_ && embedding_.read(offset, scratch_, count);
  if (!ok) ioFailed_ = true;
  return ok;
}

void Model::attention(int layer, int pos) {
  auto* history = cache_[layer];
  for (int head = 0; head < Heads; ++head) {
    const int start = head * Head;
    int8_t query[Head];
    float queryScale;
    quantize(q_ + start, Head, query, queryScale);
    quantize(k_ + start, Head, history[pos].key + start, history[pos].keyScale[head]);
    quantize(v_ + start, Head, history[pos].value + start, history[pos].valueScale[head]);
    for (int t = 0; t <= pos; ++t) {
      int sum = 0;
      for (int i = 0; i < Head; ++i) sum += query[i] * history[t].key[start + i];
      att_[t] = sum * (history[t].keyScale[head] * (queryScale / std::sqrt(float(Head))));
    }
    softmax(att_, pos + 1);
    std::fill_n(xb_ + start, Head, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      const float scale = att_[t] * history[t].valueScale[head];
      for (int i = 0; i < Head; ++i) xb_[start + i] += scale * history[t].value[start + i];
    }
  }
}

int Model::classify(uint32_t& random, Work work, float* topLogit) {
  constexpr int K = 16;
  struct Choice { float logit; int id; };
  Choice best[K];
  for (auto& c : best) c = {-std::numeric_limits<float>::infinity(), Eos};
  const auto chunk = embedding_.mapped ? 128 : int(scratchBytes_ / stride(Dim));
  for (int i = 0; i < Vocab; i += chunk) {
    if (!work.keepGoing()) return -1;
    const int rows = std::min(chunk, Vocab - i);
    const uint8_t* p;
    if (!read(Header + i * stride(Dim), rows * stride(Dim), p)) return -1;
    for (int j = 0; j < rows; ++j, p += stride(Dim)) {
      const int token = i + j;
      if (token <= Bos) continue;
      const float logit = dot(p, Dim / Group, act_, scale_, bias_);
      if (logit <= best[K - 1].logit) continue;
      int k = K - 1;
      while (k > 0 && logit > best[k - 1].logit) {
        best[k] = best[k - 1];
        --k;
      }
      best[k] = {logit, token};
    }
  }
  if (topLogit) *topLogit = best[0].logit;
  if (random == 0) return best[0].id;
  float probs[K], total = 0;
  for (int i = 0; i < K; ++i) total += (probs[i] = std::exp((best[i].logit - best[0].logit) / 0.8f));
  random ^= random << 13;
  random ^= random >> 17;
  random ^= random << 5;
  float r = total * (random >> 8) * (1.0f / 16777216);
  for (int i = 0; i < K - 1; ++i) {
    r -= probs[i];
    if (r < 0) return best[i].id;
  }
  return best[K - 1].id;
}

int Model::step(int token, int pos, bool sample, uint32_t& random, Work work,
                float* topLogit) {
  if (token < 0 || token >= Vocab || pos < 0 || pos >= context_) return -1;
  const uint8_t* row;
  if (!read(Header + token * stride(Dim), stride(Dim), row)) return -1;
  for (int i = 0; i < Dim; i += Group, row += 18) {
    const float s = weightScale(row);
    for (int j = 0; j < Group / 2; ++j) {
      x_[i + 2 * j] = ((row[j] & 15) - 8) * s;
      x_[i + 2 * j + 1] = ((row[j] >> 4) - 8) * s;
    }
  }

  for (int i = 0; i < Head / 2; ++i) {
    cos_[i] = std::cos(pos * freq_[i]);
    sin_[i] = std::sin(pos * freq_[i]);
  }

  for (int layer = 0; layer < Layers; ++layer) {
    if (!work.keepGoing()) return -1;
    const size_t d = layer * Dim * stride(Dim), h = layer * Hidden * stride(Dim);
    rms(xb_, x_, norms_ + layer * Dim);
    quantizeGroups(xb_, Dim, act_, scale_, bias_);
    matvec(q_, matrices_[0] + d, Dim, Dim, act_, scale_, bias_);
    matvec(k_, matrices_[1] + d, Dim, Dim, act_, scale_, bias_);
    matvec(v_, matrices_[2] + d, Dim, Dim, act_, scale_, bias_);

    for (int i = 0; i < Dim; i += 2) {
      const int freq = i % Head / 2;
      const float c = cos_[freq], s = sin_[freq];
      for (auto* vector : {q_, k_}) {
        const float a = vector[i], b = vector[i + 1];
        vector[i] = a * c - b * s;
        vector[i + 1] = a * s + b * c;
      }
    }
    attention(layer, pos);
    quantizeGroups(xb_, Dim, act_, scale_, bias_);
    matvec(tmp_, matrices_[3] + d, Dim, Dim, act_, scale_, bias_);
    for (int i = 0; i < Dim; ++i) x_[i] += tmp_[i];

    rms(xb_, x_, norms_ + (Layers + layer) * Dim);
    quantizeGroups(xb_, Dim, act_, scale_, bias_);
    matvec(h_, matrices_[4] + h, Dim, Hidden, act_, scale_, bias_);
    matvec(gate_, matrices_[6] + h, Dim, Hidden, act_, scale_, bias_);
    for (int i = 0; i < Hidden; ++i) h_[i] = (h_[i] / (1.0f + std::exp(-h_[i]))) * gate_[i];
    quantizeGroups(h_, Hidden, act_, scale_, bias_);
    matvec(tmp_, matrices_[5] + layer * Dim * stride(Hidden), Hidden, Dim, act_, scale_, bias_);
    for (int i = 0; i < Dim; ++i) x_[i] += tmp_[i];
  }

  if (!sample) return 0;
  rms(xb_, x_, norms_ + 2 * Layers * Dim);
  quantizeGroups(xb_, Dim, act_, scale_, bias_);
  return classify(random, work, topLogit);
}

bool Tokenizer::begin(const uint8_t* data, size_t bytes) {
  if (!data || bytes < TokBlob || std::memcmp(data, "X4TOK001", Header) != 0) return false;
  const auto maxLength = load<uint32_t>(data + Header);
  const auto blobLength = load<uint32_t>(data + Header + 4);
  if (!maxLength || maxLength > 128 || blobLength != bytes - TokBlob) return false;
  // Validate lookup bounds once before any model output or button input.
  for (int t = 0; t < Vocab; ++t) {
    const uint32_t offset = load<uint32_t>(data + TokIndex + 8 * t);
    if (offset >= blobLength || load<uint16_t>(data + TokOrder + 2 * t) >= Vocab ||
        !std::memchr(data + TokBlob + offset, 0, std::min<size_t>(maxLength + 1, blobLength - offset)))
      return false;
  }
  data_ = data;
  blob_ = reinterpret_cast<const char*>(data + TokBlob);
  return true;
}

const char* Tokenizer::piece(int token) const {
  if (!data_ || token < 0 || token >= Vocab) return "";
  return blob_ + load<uint32_t>(data_ + TokIndex + token * 8);
}

int Tokenizer::lookup(const char* text) const {
  int lo = 0, hi = Vocab;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const int id = load<uint16_t>(data_ + TokOrder + 2 * mid);
    const int cmp = std::strcmp(piece(id), text);
    if (!cmp) return id;
    if (cmp < 0) lo = mid + 1; else hi = mid;
  }
  return -1;
}

int Tokenizer::encode(const char* prompt, uint16_t* tokens, int capacity) const {
  if (!data_ || capacity < 1) return -1;
  int count = 0;
  const auto append = [&](int token) {
    if (count == capacity || token < 0) return false;
    tokens[count++] = token;
    return true;
  };
  append(Bos);
  if (*prompt && !append(lookup(" "))) return -1;
  // The on-screen keyboard emits printable ASCII.
  for (auto* c = prompt; *c; ++c) {
    const char letter[] = {*c, 0};
    const int token = lookup(letter);
    if (!append(token >= 0 ? token : int(uint8_t(*c)) + 3)) return -1;
  }
  while (true) {
    float best = -std::numeric_limits<float>::infinity();
    int at = -1, merge = -1;
    for (int i = 1; i + 1 < count; ++i) {
      char pair[257];
      const auto* a = piece(tokens[i]);
      const auto* b = piece(tokens[i + 1]);
      const auto len = std::strlen(a);
      std::memcpy(pair, a, len);
      std::strcpy(pair + len, b);
      const int id = lookup(pair);
      if (id < 0) continue;
      const float score = load<float>(data_ + TokIndex + 8 * id + 4);
      if (score > best) { at = i; merge = id; best = score; }
    }
    if (at < 0) break;
    tokens[at] = merge;
    std::memmove(tokens + at + 1, tokens + at + 2, (count - at - 2) * sizeof(*tokens));
    --count;
  }
  return count;
}
}  // namespace stories
