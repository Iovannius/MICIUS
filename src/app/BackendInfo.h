#pragma once
#include "micius_runtime.h"

namespace micius::backend {
const char* GEMVBackend(const MiciusContext*) noexcept;
const char* ScaleValidation() noexcept;
const char* ScaleStorage(const MiciusContext*) noexcept;
const char* AttentionValue() noexcept;
const char* QueryCache() noexcept;
const char* ScoreTile() noexcept;
const char* InputPreparation() noexcept;
}
