#pragma once
#include "micius_gguf.h"
#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

namespace micius::gguf {
// Shared by the dense, full-causal decoder modules. Architecture modules
// choose their own metadata prefix, semantics and tensor mapping.
inline int DescribeDecoder(const MiciusGGUFMetadataV1* view, MiciusGGUFPlanV1* out,
                           const char* prefix, uint32_t features)
{
    if (!view || !out || !view->vocab_size) return 0;
    MiciusGGUFPlanV1 plan{};
    auto& c = plan.config;
    c.vocab_size = view->vocab_size;
    c.execution_mode = MICIUS_EXECUTION_REFERENCE;
    c.kv_pool_size_mb = 16;
    const auto key = [prefix](const char* field) { return std::string(prefix) + "." + field; };
    const auto integer = [&](const char* field, uint32_t& value, bool required = true) {
        uint64_t n = 0;
        const int rc = view->get_uint(view->reader, key(field).c_str(), &n);
        if (rc == 0) return !required;
        if (rc != 1 || n > UINT32_MAX) return false;
        value = static_cast<uint32_t>(n);
        return true;
    };
    const auto positive_float = [&](const char* field, float& value, bool required) {
        double d = 0;
        const int rc = view->get_float(view->reader, key(field).c_str(), &d);
        if (rc == 0) return !required;
        if (rc != 1 || !(d > 0 && d <= std::numeric_limits<float>::max())) return false;
        value = static_cast<float>(d);
        return value > 0;
    };
    if (!integer("embedding_length", c.hidden_dim) || !integer("block_count", c.num_layers) ||
        !integer("attention.head_count", c.num_heads) || !integer("feed_forward_length", c.intermediate_dim) ||
        !integer("context_length", c.max_seq_len) || !c.hidden_dim || !c.num_heads ||
        !c.num_layers || !c.intermediate_dim || !c.max_seq_len) return 0;
    c.num_kv_heads = c.num_heads;
    c.head_dim = c.hidden_dim % c.num_heads ? 0 : c.hidden_dim / c.num_heads;
    if(!integer("attention.key_length", c.head_dim, false) || !c.head_dim)return 0;
    c.rope_dimension_count = c.head_dim;
    c.rope_freq_base = 10000.0f;
    uint32_t key_dim = c.head_dim, value_dim = c.head_dim, window = 0, vocab = c.vocab_size, experts=0, used=0;
    if (!integer("attention.head_count_kv", c.num_kv_heads, false) || !c.num_kv_heads ||
        c.num_heads % c.num_kv_heads || !integer("rope.dimension_count", c.rope_dimension_count, false) ||
        c.rope_dimension_count != c.head_dim || c.head_dim % 2 ||
        !integer("attention.key_length", key_dim, false) || key_dim != c.head_dim ||
        !integer("attention.value_length", value_dim, false) || value_dim != c.head_dim ||
        !integer("attention.sliding_window", window, false) || window ||
        !integer("vocab_size", vocab, false) || vocab != c.vocab_size ||
        !positive_float("attention.layer_norm_rms_epsilon", c.rms_norm_epsilon, true) ||
        !positive_float("rope.freq_base", c.rope_freq_base, false)) return 0;
    if(!integer("expert_count",experts,false) || experts || !integer("expert_used_count",used,false) || used)return 0;
    // A matching architecture name must not silently drop execution semantics
    // supplied by a third-party converter. Only neutral values are supported.
    const auto neutral = [&](const char* field, double expected) {
        double value = 0;
        const int found = view->get_float(view->reader, key(field).c_str(), &value);
        return found == 0 || (found == 1 && value == expected);
    };
    for (const auto* field : {"logit_scale", "residual_scale", "embedding_scale"})
        if (!neutral(field, 1.0)) return 0;
    for (const auto* field : {"attn_logit_softcapping", "final_logit_softcapping", "attention.clamp_kqv"})
        if (!neutral(field, 0.0)) return 0;
    double explicit_scale = 0;
    if (view->get_float(view->reader, key("attention.scale").c_str(), &explicit_scale) != 0) return 0;
    const char* scaling = nullptr;
    const int rc = view->get_string(view->reader, key("rope.scaling.type").c_str(), &scaling);
    if (rc < 0) return 0;
    if (rc && scaling[0] && std::string_view(scaling)!="none") {
        if (!(features & MICIUS_GGUF_ROPE_FACTORS) || std::string_view(scaling)!="llama3") return 0;
        features |= MICIUS_GGUF_REQUIRE_ROPE_FACTORS;
    }
    if (!(features & MICIUS_GGUF_REQUIRE_ROPE_FACTORS) && !neutral("rope.scaling.factor", 1.0)) return 0;
    plan.features = features;
    *out = plan;
    return 1;
}

inline const char* StandardTensor(const char* name, const MiciusGGUFPlanV1* plan)
{
    const std::string_view key(name);
    if (key == "token_embd.weight" || key == "output.weight" || key == "output_norm.weight") return name;
    if ((plan->features & MICIUS_GGUF_ROPE_FACTORS) && key=="rope_freqs.weight")return name;
    if (!key.starts_with("blk.")) return nullptr;
    const auto dot = key.find('.', 4);
    if (dot == key.npos || dot == 4 || (dot > 5 && key[4] == '0')) return nullptr;
    uint32_t layer = 0;
    const auto parsed = std::from_chars(key.data()+4, key.data()+dot, layer);
    if (parsed.ec != std::errc{} || parsed.ptr != key.data()+dot || layer >= plan->config.num_layers) return nullptr;
    const auto suffix = key.substr(dot+1);
    for (const auto* s : {"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight",
                          "attn_norm.weight","ffn_norm.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"})
        if (suffix == s) return name;
    if ((plan->features & MICIUS_GGUF_QK_NORM) && (suffix=="attn_q_norm.weight" || suffix=="attn_k_norm.weight"))return name;
    if ((plan->features & MICIUS_GGUF_QKV_BIAS) &&
        (suffix == "attn_q.bias" || suffix == "attn_k.bias" || suffix == "attn_v.bias")) return name;
    return nullptr;
}
} // namespace micius::gguf
