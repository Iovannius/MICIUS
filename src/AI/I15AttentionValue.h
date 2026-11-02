#pragma once
#include "AIComputKrnl.h"
// Private exact Value64 tiled path; unsupported cases retain I13 behavior.
AttentionValueStatus AccumulateAttentionValuesI15(const float*,double,const float*,size_t,float*,uint32_t,uint32_t) noexcept;
