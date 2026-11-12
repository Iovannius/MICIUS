#pragma once
#include <filesystem>
#include <span>
#include <vector>
#include <cstdint>
#include <string>
namespace micius::session {
std::string Utf8(const std::filesystem::path& path);
std::vector<uint8_t> ReadFile(const std::filesystem::path& path,size_t limit);
void AtomicWrite(const std::filesystem::path& path,std::span<const uint8_t> bytes);
void AtomicText(const std::filesystem::path& path,const std::string& text);
void PlainDirectory(const std::filesystem::path& path);
class Lock {
public:
    explicit Lock(const std::filesystem::path& path);
    ~Lock();
    Lock(const Lock&)=delete;
    Lock& operator=(const Lock&)=delete;
private:
    intptr_t handle=-1;
};
}
