#include "AICoreEngine.h"
#include "micius_state.h"
#include "AI/FileWeights.h"
#include "session/Binary.h"
#include <cmath>
#include <immintrin.h>

namespace {
using namespace micius::state;
void Shape(Writer& w,const MiciusConfig& c,uint32_t split,bool bias) {
    for (uint32_t v : {c.hidden_dim,c.vocab_size,c.max_seq_len,c.num_layers,c.num_heads,
        c.num_kv_heads,c.head_dim,c.intermediate_dim,c.execution_mode,c.rope_dimension_count}) w.U32(v);
    w.Float(c.rms_norm_epsilon); w.Float(c.rope_freq_base);
    w.U32(split); w.U32(bias);
}
void Require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
}

size_t AIEngineFront::ReferenceStateBytes() const
{
    Require(!speculative_enabled && model_bound && reference_k_cache && reference_v_cache,
        "Snapshot needs a bound single-thread model");
    Require(!reference_seq_len || reference_has_logits,"Cannot save incomplete forward state");
    size_t kv=Product(Product(Product(reference_seq_len,reference_kv_dim),config.num_layers),8);
    size_t rest=148+(reference_has_logits?Product(config.vocab_size,4):0);
    Require(kv<=SIZE_MAX-rest,"Snapshot size overflow");return kv+rest;
}
std::vector<uint8_t> AIEngineFront::ExportReferenceState() const
{
    Writer w;
    w.bytes.reserve(ReferenceStateBytes());
    w.Raw("MCSKV001",8); w.Text(Micius_StateContractV1());
    Shape(w,config,uint32_t(model_layers[0].split_half_rope) | (file_weights ? (2U | (file_weights->qk_norm?4U:0U) | (file_weights->rope_factors?8U:0U)) : 0U),model_layers[0].q_bias!=nullptr);
    w.U32(_mm_getcsr() & 0xffc0u); // control/masks, excluding sticky exception flags
    w.U32(reference_seq_len); w.U32(reference_has_logits);
    w.U32(total_tokens_generated.load(std::memory_order_relaxed));
    const size_t row=Product(reference_seq_len,reference_kv_dim);
    for (uint32_t layer=0;layer<config.num_layers;++layer) {
        size_t offset=size_t(layer)*config.max_seq_len*reference_kv_dim;
        for (const float* p : {reference_k_cache.get()+offset,reference_v_cache.get()+offset})
            for(size_t i=0;i<row;++i) { Require(std::isfinite(p[i]),"Nonfinite KV"); w.Float(p[i]); }
    }
    if(reference_has_logits) for(size_t i=0;i<config.vocab_size;++i) {
        Require(std::isfinite(reference_logits[i]),"Nonfinite logits"); w.Float(reference_logits[i]);
    }
    return std::move(w.bytes);
}

void AIEngineFront::ImportReferenceState(std::span<const uint8_t> bytes,uint32_t expected_length)
{
    Require(!speculative_enabled && model_bound && reference_k_cache && reference_v_cache,
        "Snapshot needs a bound single-thread model");
    Reader r{bytes};
    auto magic=r.Raw(8);
    Require(std::memcmp(magic.data(),"MCSKV001",8)==0,"Unknown KV snapshot format");
    Require(r.Text()==Micius_StateContractV1(),"Snapshot numerical/build contract differs");
    Writer shape; Shape(shape,config,uint32_t(model_layers[0].split_half_rope) | (file_weights ? (2U | (file_weights->qk_norm?4U:0U) | (file_weights->rope_factors?8U:0U)) : 0U),model_layers[0].q_bias!=nullptr);
    auto actual=r.Raw(shape.bytes.size());
    Require(std::equal(actual.begin(),actual.end(),shape.bytes.begin()),"Snapshot model shape differs");
    Require(r.U32()==(_mm_getcsr() & 0xffc0u),"Snapshot floating-point control differs");
    const uint32_t length=r.U32(), valid=r.U32(), generated=r.U32();
    Require(length==expected_length && length<=config.max_seq_len && valid==uint32_t(length!=0),"Invalid snapshot sequence length");
    const size_t row=Product(length,reference_kv_dim), values=Product(Product(row,config.num_layers),2);
    const size_t floats=values+(valid?config.vocab_size:0);
    Require(bytes.size()-r.pos==Product(floats,4),"Snapshot payload length differs");
    const size_t start=r.pos;
    for(size_t i=0;i<floats;++i) Require(std::isfinite(r.Float()),"Nonfinite snapshot payload");
    r.End();
    // Validation is complete. The following commit uses preallocated storage,
    // performs no inference/allocation, and cannot leave a partial replacement.
    Require(ResetState(),"Engine cannot reset for restore");
    r.pos=start;
    for(uint32_t layer=0;layer<config.num_layers;++layer) {
        size_t offset=size_t(layer)*config.max_seq_len*reference_kv_dim;
        for(float* p : {reference_k_cache.get()+offset,reference_v_cache.get()+offset})
            for(size_t i=0;i<row;++i)p[i]=r.Float();
    }
    if(valid)for(size_t i=0;i<config.vocab_size;++i)reference_logits[i]=r.Float();
    reference_seq_len=length; reference_has_logits=valid!=0;
    total_tokens_generated.store(generated,std::memory_order_relaxed);
}
