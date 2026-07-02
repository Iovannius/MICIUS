#include "MiciusShell.h"

#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
    #ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
    #define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
    #endif
#endif

MiciusShell::MiciusShell()
{
    InitConsole();
    BuildDictionaries();

    ctx.engine_ctx = nullptr; 
    ctx.prompt_text.assign("micius> ");
    ctx.active_model.assign("");
    ctx.should_exit = false;
    ctx.is_logging = false;
}

MiciusShell::~MiciusShell()
{
    if (ctx.is_logging && ctx.log_file.is_open()) {
        ctx.log_file.close();
    }
    if (ctx.engine_ctx) {
        Micius_DestroyEngine(ctx.engine_ctx);
    }
}

void MiciusShell::InitConsole()
{
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif
}

void MiciusShell::BuildDictionaries()
{
    /*命令路由*/
    cmd_router.insert({"help", {"Show all available commands", CmdHelp}});
    cmd_router.insert({"exit", {"Terminate the runtime shell", CmdExit}});
    cmd_router.insert({"quit", {"Terminate the runtime shell (alias)", CmdExit}});
    cmd_router.insert({"load", {"Load a GGUF model", CmdLoad}});
    cmd_router.insert({"sandbox", {"Mount a stress testing sandbox", CmdSandbox}});
    cmd_router.insert({"stats", {"Show telemetry and hardware bounds", CmdStats}});
    cmd_router.insert({"debug", {"Debug underlying mechanics (e.g. debug workers)", CmdDebugWorkers}});
    cmd_router.insert({"clear", {"Clear the console screen", CmdClear}});
    cmd_router.insert({"cls", {"Clear the console screen (alias)", CmdClear}});
    cmd_router.insert({"log", {"Archive session (usage: log start <file> | log stop)", CmdLog}});

    /*字符串字典*/
    msg_dict.insert({"err_no_engine", "\033[1;31m[Error]\033[0m Engine offline. Please load a model or sandbox first.\n"});
    msg_dict.insert({"err_usage_load", "Usage: load <model_path.gguf>\n"});
    msg_dict.insert({"err_usage_sandbox", "Usage: sandbox <name>\n"});
    msg_dict.insert({"err_usage_log", "Usage: log start <filename> OR log stop\n"});
    msg_dict.insert({"msg_exit", "[Shell] Halting MICIUS Engine...\n"});
    msg_dict.insert({"msg_sandbox_lock", "[Sandbox] Array locked. Awaiting execution.\n"});
    msg_dict.insert({"msg_unknown_cmd", "Unknown command. Type 'help' for a list.\n"});
}

const std::string& MiciusShell::GetMsg(const std::string& key)
{
    auto it = msg_dict.find(key);
    if (it != msg_dict.end()) return it->second;
    static const std::string empty = "";
    return empty;
}

std::string MiciusShell::StripANSI(const std::string& input)
{
    std::string output;
    bool in_escape = false;
    for (char c : input) {
        if (c == '\033') in_escape = true;
        else if (in_escape) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) in_escape = false;
        }
        else output += c;
    }
    return output;
}

//Shell文本打印走这里
void MiciusShell::ShellPrint(const std::string& msg)
{
    std::cout << msg;
    if (ctx.is_logging && ctx.log_file.is_open()) {
        ctx.log_file << StripANSI(msg);
        ctx.log_file.flush();
    }
}

std::vector<std::string> MiciusShell::TokenizeInput(const std::string& input)
{
    std::vector<std::string> tokens;
    std::stringstream ss(input);
    std::string token;
    while (ss >> token) tokens.push_back(token);
    return tokens;
}

void MiciusShell::RunREPL()
{
    std::string input_line;
    while (!ctx.should_exit) {
        std::cout << "\033[1;36m" << ctx.prompt_text << "\033[0m" << std::flush;
        
        if (!std::getline(std::cin, input_line)) break; 
        if (input_line.empty()) continue;
        
        //如果开启了日志，原样存档用户输入的命令（清除提示符的颜色代码）
        if (ctx.is_logging && ctx.log_file.is_open()) {
            ctx.log_file << StripANSI(ctx.prompt_text) << input_line << "\n";
            ctx.log_file.flush();
        }

        std::vector<std::string> args = TokenizeInput(input_line);
        if (args.empty()) continue;
        
        const std::string& cmd_name = args[0];
        
        //从哈希表中提取函数指针
        auto it = cmd_router.find(cmd_name);
        if (it != cmd_router.end()) {
            it->second.handler(this, &ctx, args);
        } else {
            ShellPrint(GetMsg("msg_unknown_cmd"));
        }
    }
}

//============================================================================
//命令回调区
//============================================================================

void MiciusShell::CmdHelp(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    std::stringstream ss;
    ss << "\nAvailable Commands:\n--------------------------------------------------------\n";
    for (const auto& pair : shell->cmd_router) {
        ss << "  " << pair.first;
        //简单的对齐填充
        int padding = 15 - (int)pair.first.length();
        if (padding > 0) ss << std::string(padding, ' ');
        ss << pair.second.description << "\n";
    }
    ss << "--------------------------------------------------------\n\n";
    shell->ShellPrint(ss.str());
}

void MiciusShell::CmdClear(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    //发送清屏并复位光标
    shell->ShellPrint("\033[2J\033[1;1H");
}

void MiciusShell::CmdLog(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (args.size() < 2) { shell->ShellPrint(shell->GetMsg("err_usage_log")); return; }
    
    if (args[1].compare("start") == 0) {
        if (args.size() < 3) { shell->ShellPrint(shell->GetMsg("err_usage_log")); return; }
        if (ctx->is_logging) { shell->ShellPrint("Logging is already active.\n"); return; }
        
        ctx->log_file.open(args[2], std::ios::app);
        if (!ctx->log_file.is_open()) {
            shell->ShellPrint("\033[1;31m[Error]\033[0m Failed to open log file.\n");
        } else {
            ctx->is_logging = true;
            std::string msg;
            msg.assign("[System] Archiving started to: ").append(args[2]).append("\n");
            shell->ShellPrint(msg);
        }
    } 
    else if (args[1].compare("stop") == 0) {
        if (!ctx->is_logging) { shell->ShellPrint("Logging is not active.\n"); return; }
        ctx->log_file.close();
        ctx->is_logging = false;
        shell->ShellPrint("[System] Archiving stopped.\n");
    } 
    else { shell->ShellPrint(shell->GetMsg("err_usage_log")); }
}

void MiciusShell::CmdExit(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    shell->ShellPrint(shell->GetMsg("msg_exit"));
    ctx->should_exit = true;
}

void MiciusShell::CmdLoad(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (args.size() < 2) { shell->ShellPrint(shell->GetMsg("err_usage_load")); return; }
    std::string file_name = args[1];
    
    std::string msg_parse;
    msg_parse.assign("[Loader] Attempting to parse ").append(file_name).append("...\n");
    shell->ShellPrint(msg_parse);
    shell->ShellPrint("[Loader] Quant: INT8 | Hidden: 4096 | Layers: 32\n");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    
    //使用char类型的'.'避免隐式转换报错
    size_t last_dot = file_name.find_last_of('.');
    std::string model_name = (last_dot == std::string::npos) ? file_name : file_name.substr(0, last_dot);
    
    ctx->active_model.assign(model_name);
    ctx->prompt_text.assign(model_name).append("> ");
    
    std::string msg_mount;
    msg_mount.assign("[Loader] \033[1;32m").append(model_name).append("\033[0m securely mounted to Engine.\n");
    shell->ShellPrint(msg_mount);
}

void MiciusShell::CmdSandbox(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (args.size() < 2) { shell->ShellPrint(shell->GetMsg("err_usage_sandbox")); return; }
    std::string sandbox_name = args[1];
    shell->ShellPrint("[Sandbox] Igniting internal stress matrix...\n");
    
    ctx->active_model.assign(sandbox_name);
    ctx->prompt_text.assign("sandbox:").append(sandbox_name).append("> ");
    shell->ShellPrint(shell->GetMsg("msg_sandbox_lock"));
}

void MiciusShell::CmdStats(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (!ctx->engine_ctx) { shell->ShellPrint(shell->GetMsg("err_no_engine")); return; }
    
    MiciusRuntimeStats stats;
    Micius_GetRuntimeStats(ctx->engine_ctx, &stats);
    uint32_t worker_count = Micius_GetWorkerCount(ctx->engine_ctx);
    
    float kv_usage = (stats.kv_slots_total > 0) ? ((float)stats.kv_slots_used / stats.kv_slots_total * 100.0f) : 0.0f;

    std::stringstream ss;
    ss << "\n========== MICIUS Runtime Telemetry ==========\n";
    ss << " Tokens Generated : " << stats.total_tokens_generated << "\n";
    ss << " SpecTree Rejects : " << stats.tree_reject_count << " (Hostile Churn)\n";
    ss << " KV Pool Usage    : [";
    int bar_width = 20, filled = (int)(kv_usage / 100.0f * bar_width);
    for (int i = 0; i < bar_width; ++i) ss << (i < filled ? "#" : "-");
    ss << "] " << stats.kv_slots_used << "/" << stats.kv_slots_total << " (" << (int)kv_usage << "%)\n";
    ss << " Worker Array     : " << worker_count << " Active Threads\n==============================================\n\n";
    
    shell->ShellPrint(ss.str());
}

void MiciusShell::CmdDebugWorkers(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (args.size() < 2 || args[1].compare("workers") != 0) {
        shell->ShellPrint("Usage: debug workers\n");
        return;
    }
    if (!ctx->engine_ctx) { shell->ShellPrint(shell->GetMsg("err_no_engine")); return; }

    uint32_t worker_count = Micius_GetWorkerCount(ctx->engine_ctx);
    std::stringstream ss;
    ss << "\n[Worker Array Workload Distribution]\n";
    for (uint32_t i = 0; i < worker_count; ++i) {
        MiciusWorkerInfo info;
        if (Micius_GetWorkerInfo(ctx->engine_ctx, i, &info)) {
            ss << " Worker " << i << " (Core " << info.core_id << "):\n";
            ss << "   - Local Processed : " << info.tasks_processed << "\n";
            ss << "   - Xorshift Stolen : " << info.tasks_stolen << "\n";
        }
    }
    ss << "\n";
    shell->ShellPrint(ss.str());
}