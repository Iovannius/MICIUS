// Optional reference tool for regenerating M2 fixtures. Link only with the
// pinned llama.cpp 0.6.0 libraries and its bundled Jinja sources, never MICIUS.
#include "llama.h"
#include "json.h"
#include "jinja/parser.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv)
{
    if (argc!=5) return 2; // MODEL ordinary|messages INPUT OUTPUT_U32
    try {
        auto params=llama_model_default_params();params.vocab_only=true;params.n_gpu_layers=0;
        llama_backend_init();
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],params),llama_model_free);
        if (!model) throw std::runtime_error("vocabulary open failed");
        const auto* vocab=llama_model_get_vocab(model.get());
        std::ifstream in(argv[3],std::ios::binary);
        if (!in) throw std::runtime_error("input open failed");
        std::string text{std::istreambuf_iterator<char>(in),{}};
        bool special=false;
        if (std::string(argv[2])=="messages") {
            const std::string tmpl=llama_model_chat_template(model.get(),nullptr);
            jinja::lexer lexer;
            auto tokens=lexer.tokenize(tmpl);
            auto program=jinja::parse_from_tokens(tokens);
            jinja::context ctx(tmpl);
            jinja::global_from_json(ctx,common_json::parse(text),true);
            jinja::runtime runtime(ctx);
            text=jinja::render_string_parts(runtime.gather_string_parts(runtime.execute(program)));
            special=true;
        } else if (std::string(argv[2])!="ordinary") return 2;
        int size=llama_tokenize(vocab,text.data(),int32_t(text.size()),nullptr,0,false,special);
        std::vector<llama_token> ids(size<0 ? -size : size);
        int count=llama_tokenize(vocab,text.data(),int32_t(text.size()),ids.data(),int32_t(ids.size()),false,special);
        if (count!=int(ids.size())) throw std::runtime_error("tokenization failed");
        std::ofstream out(argv[4],std::ios::binary);
        out.write(reinterpret_cast<const char*>(ids.data()),count*4);
        out.close();return out ? 0 : 1;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
