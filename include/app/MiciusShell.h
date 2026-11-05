#pragma once

#include "../micius_runtime.h"
#include "micius_gguf.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <csignal>
#include <atomic>

class MiciusShell;

//Shell运行时的独立上下文
struct ShellContext {
    MiciusContext* engine_ctx = nullptr; //引擎不透明句柄
    std::string prompt_text;   //动态提示符文本
    std::string active_model;  //当前挂载的模型或沙箱名称
    bool should_exit = false;  //REPL退出标志

    //存档与日志
    bool is_logging = false;
    std::ofstream log_file;

    std::string model_dir; //默认模型目录
    std::string log_dir;   //默认日志目录

    //引擎动态配置与异步控制
    MiciusConfig engine_config{};            //当前挂载引擎的配置参数
    std::atomic<bool> is_interrupted{false}; //软中断安全标志

    MiciusFileModel* file_model = nullptr; // owns engine_ctx and its mapped weights

};

//命令处理函数签名
typedef void (*CommandHandler)(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);

//命令路由条目定义
struct CommandDef {
    std::string description;
    CommandHandler handler;
};

class MiciusShell
{

private:
    ShellContext ctx;
    // Private experiment entry; no public runtime ABI or default CLI change.
    int RunI6Benchmark(const std::vector<std::string>& args);

    //指令路由表与字符串字典
    std::unordered_map<std::string, CommandDef> cmd_router;
    std::unordered_map<std::string, std::string> msg_dict;

    //初始化与注册
    void InitConsole();
    void SetupEnvironment();
    void BuildDictionaries();
    std::vector<std::string> TokenizeInput(const std::string& input);

    //释放由Shell管理的动态内存池
    static void ReleaseFile(ShellContext* ctx);
    bool LoadFile(const std::string& filename, uint32_t context_length = 0);
    void LogText(const std::string& plain, bool flush);
    void LogInput(const std::string& prompt, const std::string& line);
    bool ReadLine(std::string& line);
#ifdef MICIUS_HAS_TEXT
    static void CmdChat(MiciusShell*, ShellContext*, const std::vector<std::string>&);
    int ChatLoop(uint32_t max_new, bool trace = false);
#endif

    //过滤器，去除控制字符与彩色字符
    std::string StripANSI(const std::string& input);

    //全局信号捕获回调
    static void SignalHandler(int signum);

    //基础命令静态回调
    static void CmdHelp(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdExit(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdLoad(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdStats(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdDebugWorkers(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    
    static void CmdClear(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdLog(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);

    static void CmdGenerate(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);

public:
    MiciusShell();
    ~MiciusShell();

    //控制台重定向输出端口
    void ShellPrint(const std::string& msg);
    const std::string& GetMsg(const std::string& key);

    //交互主循环
    void RunREPL();
#ifdef MICIUS_HAS_TEXT
    int RunChat(const std::vector<std::string>& args);
#endif

    // Non-interactive real-model timing entry. Model parsing, tensor conversion,
    // binding, reset, prompt ingestion and warmup remain outside the timed ROI.
    int RunModelBenchmark(const std::vector<std::string>& args);
};
