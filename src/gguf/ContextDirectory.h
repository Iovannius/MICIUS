#pragma once
#include <cstddef>
#include <string>

namespace micius::gguf {
std::string Sha256(const void* bytes, size_t size);
std::string CreateContextDirectory(const char* root, const void* bytes, size_t size);
}
