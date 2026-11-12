#include "ContextDirectory.h"
#include "NativePath.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/sha.h>
#endif

namespace micius::gguf {
namespace fs = std::filesystem;

static std::string Hex(const unsigned char* bytes, size_t size)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(size * 2);
    for (size_t i = 0; i < size; ++i) {
        text.push_back(digits[bytes[i] >> 4]);
        text.push_back(digits[bytes[i] & 15]);
    }
    return text;
}

std::string Sha256(const void* data, size_t size)
{
    std::array<unsigned char, 32> digest{};
    auto bytes = static_cast<const unsigned char*>(data);
#ifdef _WIN32
    struct Handles {
        BCRYPT_ALG_HANDLE alg{};
        BCRYPT_HASH_HANDLE hash{};
        ~Handles() {
            if (hash) BCryptDestroyHash(hash);
            if (alg) BCryptCloseAlgorithmProvider(alg, 0);
        }
    } h;
    if (BCryptOpenAlgorithmProvider(&h.alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptCreateHash(h.alg, &h.hash, nullptr, 0, nullptr, 0, 0) < 0)
        throw std::runtime_error("Cannot initialize SHA-256 for model identity");
    while (size) {
        const auto chunk = static_cast<ULONG>(std::min<size_t>(size, std::numeric_limits<ULONG>::max()));
        if (BCryptHashData(h.hash, const_cast<PUCHAR>(bytes), chunk, 0) < 0)
            throw std::runtime_error("Cannot hash the mapped model");
        bytes += chunk;
        size -= chunk;
    }
    if (BCryptFinishHash(h.hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0)
        throw std::runtime_error("Cannot finish the model digest");
#else
    if (!SHA256(bytes, size, digest.data())) throw std::runtime_error("Cannot hash the mapped model");
#endif
    return Hex(digest.data(), digest.size());
}

std::string CreateContextDirectory(const char* root, const std::string& identity)
{
    const auto parent = fs::absolute(fs::u8path(root)) / identity / "contexts";
    fs::create_directories(NativePath(parent));
    std::random_device random;
    std::uniform_int_distribution<unsigned> sample(0, 255);
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        std::array<unsigned char, 16> nonce{};
        for (auto& v : nonce) v = static_cast<unsigned char>(sample(random));
        const auto dir = parent / Hex(nonce.data(), nonce.size());
        const auto kv = dir / "kv";
        const auto u8 = dir.u8string();
        std::string result(u8.begin(), u8.end());
        if (!fs::create_directory(NativePath(dir))) continue;
        try {
            if (!fs::create_directory(NativePath(kv))) throw std::runtime_error("KV directory already exists");
        } catch (...) {
            std::error_code ignored;
            fs::remove(NativePath(dir), ignored); // only the empty directory reserved by this call
            throw;
        }
        return result;
    }
    throw std::runtime_error("Cannot reserve a unique context directory");
}
}
