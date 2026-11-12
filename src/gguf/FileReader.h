#pragma once
#include "micius_gguf.h"
class GGUFParser;
namespace micius::gguf {
// The reader and the already bound weights belong to the same file session.
bool AcquireChatEngine(MiciusFileModel* model);
void ReleaseChatEngine(MiciusFileModel* model) noexcept;
const GGUFParser& FileReader(const MiciusFileModel& model);
}
