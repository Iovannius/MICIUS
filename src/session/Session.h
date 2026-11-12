#pragma once
#include "text/Conversation.h"
#include "SamplingConfig.h"
#include "Storage.h"
#include "micius_session.h"
#include <memory>

namespace micius::session {
class EngineLease {
    MiciusFileModel* file;
public:
    explicit EngineLease(MiciusFileModel* f);
    ~EngineLease();
    EngineLease(const EngineLease&)=delete;
};
class Session {
public:
    Session(MiciusFileModel* file,const char* id);
    const std::string& Directory() const { return directory; }
    std::string Configuration() const { return WriteConfig(config); }
    uint64_t Turns() const { return turns; }
    void Save();
    void Reload();
    void Reset();
    text::Answer Reply(std::string_view text,uint32_t limit,const std::atomic<bool>& cancel,
                       const std::function<void(std::string_view)>& emit);
    static std::string List(MiciusFileModel* file);
private:
    EngineLease lease;
    MiciusFileModel* file;
    std::string directory;
    std::filesystem::path path;
    std::unique_ptr<Lock> lock;
    text::Tokenizer tokenizer;
    text::Conversation chat;
    Config config;
    uint64_t turns=0;
    void Restore();
};
}
