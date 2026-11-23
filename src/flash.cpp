#include "flash.h"

#include <esp_ota_ops.h>
#include <esp_partition.h>

namespace stories {
namespace {
const esp_partition_t* embeddingPart = nullptr;
esp_partition_mmap_handle_t layerMap, tokenMap, embeddingMap;

bool readEmbedding(size_t offset, void* data, size_t count) {
  return esp_partition_read(embeddingPart, offset, data, count) == ESP_OK;
}

const uint8_t* map(const esp_partition_t* part, size_t offset, size_t length,
                   esp_partition_mmap_handle_t& handle) {
  const void* data = nullptr;
  if (esp_partition_mmap(part, offset, length, ESP_PARTITION_MMAP_DATA, &data, &handle) != ESP_OK)
    return nullptr;
  return static_cast<const uint8_t*>(data);
}
}
bool openFlash(const uint8_t*& layers, Reader& embedding, const uint8_t*& tokenizer) {
  static_assert(LayersOffset + LayerBytes <= TokenizerOffset &&
                TokenizerOffset + TokenizerBytes <= SlotSize && EmbeddingBytes <= SlotSize,
                "flash assets must fit");
  const auto* running = esp_ota_get_running_partition();
  const auto* other = esp_partition_find_first(ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_APP_OTA_1, nullptr);
  if (!running || running->address != App0Address || running->size != SlotSize ||
      !other || other->address != App1Address || other->size != SlotSize) return false;
  layers = map(running, LayersOffset, LayerBytes, layerMap);
  tokenizer = map(running, TokenizerOffset, TokenizerBytes, tokenMap);
  if (!layers || !tokenizer) return false;
  embeddingPart = other;
  // The C3's 8 MiB shared MMU window is smaller than the whole model.
  embedding = {map(other, 0, EmbeddingBytes, embeddingMap), readEmbedding};
  return true;
}
}
