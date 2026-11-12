#pragma once
#include "micius_sampling.h"
#include <cstdint>
#include <string>
#include <string_view>
namespace micius::session {
struct Config {
    MiciusSamplingOptionsV1 sample{MICIUS_SAMPLING_ABI_V1,sizeof(MiciusSamplingOptionsV1),0,1,1,0,0};
    uint64_t seed=0;
    uint32_t max_new_tokens=128;
};
Config ParseConfig(std::string_view text);
std::string WriteConfig(const Config& config);
}
