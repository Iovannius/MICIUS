#pragma once

#include "../micius_runtime.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

class MiciusShell;

//Shell运行时的独立上下文
struct ShellContext {
    MiciusContext* engine_ctx; //引擎不透明句柄
    std::string prompt_text;   //动态提示符文本
    std::string active_model;  //当前挂载的模型或沙箱名称
    bool should_exit;          //REPL退出标志

    //存档与日志
    bool is_logging;
    std::ofstream log_file;
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

    //指令路由表与字符串字典
    std::unordered_map<std::string, CommandDef> cmd_router;
    std::unordered_map<std::string, std::string> msg_dict;

    //初始化与注册
    void InitConsole();
    void BuildDictionaries();
    std::vector<std::string> TokenizeInput(const std::string& input);

    //过滤器，去除控制字符与彩色字符
    std::string StripANSI(const std::string& input);

    //基础命令静态回调
    static void CmdHelp(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdExit(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdLoad(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdSandbox(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdStats(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdDebugWorkers(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    
    static void CmdClear(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);
    static void CmdLog(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args);

public:
    MiciusShell();
    ~MiciusShell();

    void ShellPrint(const std::string& msg);
    const std::string& GetMsg(const std::string& key);

    //交互主循环
    void RunREPL();
};