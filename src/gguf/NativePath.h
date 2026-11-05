#pragma once
#include <filesystem>

namespace micius::gguf {
inline std::filesystem::path NativePath(const std::filesystem::path& path)
{
#ifdef _WIN32
    const auto absolute = std::filesystem::absolute(path).lexically_normal().make_preferred().native();
    if (absolute.starts_with(LR"(\\?\)")) return absolute;
    if (absolute.starts_with(LR"(\\)")) return LR"(\\?\UNC\)" + absolute.substr(2);
    return LR"(\\?\)" + absolute;
#else
    return path;
#endif
}
}
