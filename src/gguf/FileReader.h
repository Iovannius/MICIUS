#pragma once
#include "micius_gguf.h"
class GGUFParser;
namespace micius::gguf {
// The reader and the already bound weights belong to the same file session.
const GGUFParser& FileReader(const MiciusFileModel& model);
}
