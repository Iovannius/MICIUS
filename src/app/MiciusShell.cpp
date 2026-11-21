#include "BackendInfo.h"
#include "MiciusShell.h"
#include "gguf/NativePath.h"
#include "I22Memory.h"
#include <cstdlib>
#include <memory>
#if defined(MICIUS_I4_EVALUATION)
#include "I4Evaluation.h"
#endif
#include "BenchmarkIttRoi.h"
#include "AI/DecodeProfile.h"
#include "AI/PreparedGEMVWeights.h"
#include <fstream>

#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <bit>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <string_view>
#include <immintrin.h> //仅用于_mm_malloc与释放

#ifdef _WIN32
    #include <windows.h>
    #ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
    #define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
    #endif
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
#endif

static int CurrentProcessorForBenchmark() noexcept
{
#if defined(_WIN32)
    return static_cast<int>(GetCurrentProcessorNumber());
#elif defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

static bool PinBenchmarkThread(uint32_t cpu_id, int* observed_cpu, std::string* error)
{
    if (!observed_cpu || !error) return false;
#if defined(_WIN32)
    constexpr uint32_t kAffinityBits =
        static_cast<uint32_t>(std::numeric_limits<DWORD_PTR>::digits);
    if (cpu_id >= kAffinityBits) {
        *error = "--cpu exceeds the current Windows processor group mask";
        return false;
    }
    const DWORD_PTR mask = DWORD_PTR{1} << cpu_id;
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        *error = "SetThreadAffinityMask failed with Win32 error " +
                 std::to_string(GetLastError());
        return false;
    }
#elif defined(__linux__)
    if (cpu_id >= static_cast<uint32_t>(CPU_SETSIZE)) {
        *error = "--cpu exceeds CPU_SETSIZE";
        return false;
    }
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(static_cast<int>(cpu_id), &affinity);
    const int status = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
    if (status != 0) {
        *error = "pthread_setaffinity_np failed: " + std::string(std::strerror(status));
        return false;
    }
#else
    (void)cpu_id;
    *error = "--cpu is unsupported on this platform";
    return false;
#endif
    for (uint32_t attempt = 0; attempt < 1024U; ++attempt) {
        *observed_cpu = CurrentProcessorForBenchmark();
        if (*observed_cpu >= 0 && static_cast<uint32_t>(*observed_cpu) == cpu_id) {
            return true;
        }
        std::this_thread::yield();
    }
    *error = "thread affinity request did not take effect";
    return false;
}

static bool ParseBenchmarkU64(std::string_view text, uint64_t* value) noexcept
{
    if (!value || text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, *value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

static uint64_t HashBenchmarkWord(uint64_t hash, uint32_t value) noexcept
{
    constexpr uint64_t kPrime = UINT64_C(1099511628211);
    for (uint32_t byte = 0; byte < 4U; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8U));
        hash *= kPrime;
    }
    return hash;
}

static std::string BenchmarkApiError(const char* operation)
{
    std::ostringstream stream;
    stream << operation << " failed (code="
           << static_cast<int>(Micius_GetLastErrorCode()) << "): ";
    const char* message = Micius_GetLastError();
    stream << (message && message[0] != '\0' ? message : "no diagnostic");
    return stream.str();
}

//用于信号处理函数的静态全局指针，指向当前激活的Shell实例
static MiciusShell* g_active_shell = nullptr;

MiciusShell::MiciusShell()
{
    g_active_shell = this;

    //注册软中断拦截
    std::signal(SIGINT, MiciusShell::SignalHandler);

    InitConsole();
    SetupEnvironment();
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
    
    ReleaseFile(&ctx);

    if (g_active_shell == this) g_active_shell = nullptr;
}

void MiciusShell::ReleaseFile(ShellContext* ctx)
{
    Micius_CloseGGUFV1(ctx->file_model);
    ctx->file_model = nullptr;
    ctx->engine_ctx = nullptr;
    ctx->engine_config = {};
}

void MiciusShell::InitConsole()
{
#ifdef _WIN32
    //设置Windows控制台字符编码模式
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
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

//运行环境初始化
void MiciusShell::SetupEnvironment()
{
    ctx.model_dir = "models";
    ctx.log_dir = "logs";

    //如果不存在，创建所需的目录
    std::error_code ec;
    std::filesystem::create_directories(ctx.model_dir, ec);
    std::filesystem::create_directories(ctx.log_dir, ec);
}

//软中断信号回调
void MiciusShell::SignalHandler(int signum)
{
#ifdef _WIN32
    // The Windows CRT resets this disposition after delivering SIGINT.
    if (signum==SIGINT) std::signal(SIGINT,MiciusShell::SignalHandler);
#endif
    if (signum == SIGINT && g_active_shell != nullptr) {
        //写入中断标志而不是杀进程
        g_active_shell->ctx.is_interrupted.store(true, std::memory_order_relaxed);
    }
}

void MiciusShell::BuildDictionaries()
{
    /*命令路由*/
    cmd_router.insert({"help", {"Show all available commands", CmdHelp}});
    cmd_router.insert({"exit", {"Terminate the runtime shell", CmdExit}});
    cmd_router.insert({"load", {"Load a GGUF model", CmdLoad}});
    cmd_router.insert({"stats", {"Show telemetry and hardware bounds", CmdStats}});
    cmd_router.insert({"debug", {"Debug underlying mechanics (e.g. debug workers)", CmdDebugWorkers}});
    cmd_router.insert({"clear", {"Clear the console screen", CmdClear}});
    cmd_router.insert({"log", {"Archive session (usage: log start <file> | log stop)", CmdLog}});
    cmd_router.insert({"generate", {"Start model generation loop (Ctrl+C to interrupt)", CmdGenerate}});
#ifdef MICIUS_HAS_TEXT
    cmd_router.insert({"chat", {"UTF-8 chat: chat [max_reply_tokens], chat list, chat resume ID [max_reply_tokens]", CmdChat}});
#endif

    /*字符串字典*/
    msg_dict.insert({"err_no_engine", "\033[1;31m[Error]\033[0m Engine offline. Please load a model or sandbox first.\n"});
    msg_dict.insert({"err_usage_load", "Usage: load <model_path.gguf>\n"});
    msg_dict.insert({"err_usage_log", "Usage: log start <filename> OR log stop\n"});
    msg_dict.insert({"msg_exit", "[Shell] Halting MICIUS Engine...\n"});
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
    for (size_t i=0; i<input.size();) {
        const auto c=static_cast<unsigned char>(input[i++]);
        if (c==0x1b && i<input.size()) {
            const char kind=input[i++];
            if (kind=='[') {
                while (i<input.size()) {
                    const auto v=static_cast<unsigned char>(input[i++]);
                    if (v>=0x40 && v<=0x7e) break;
                }
            } else if (kind==']') {
                while (i<input.size()) {
                    if (input[i++]=='\a') break;
                    if (input[i-1]=='\033' && i<input.size() && input[i]=='\\') { ++i; break; }
                }
            }
        } else if ((c>=32 && c!=127) || c=='\n' || c=='\t') output+=static_cast<char>(c);
    }
    return output;
}

void MiciusShell::LogText(const std::string& plain, bool flush)
{
    if (!ctx.is_logging) return;
    ctx.log_file << plain;
    if (flush) ctx.log_file.flush();
    if (!ctx.log_file) {
        ctx.is_logging=false;
        ctx.log_file.close();
        std::cerr << "[Log] Session log write failed; logging disabled.\n";
    }
}

void MiciusShell::LogInput(const std::string& prompt, const std::string& line)
{
    if (ctx.is_logging) LogText(StripANSI(prompt+line)+"\n",true);
}

bool MiciusShell::ReadLine(std::string& line)
{
#ifdef _WIN32
    // Windows console input is UTF-16. Mintty, redirected files and pipes
    // supply UTF-8 bytes through stdin and take the ordinary getline path.
    HANDLE handle=GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode=0;
    if (GetConsoleMode(handle,&mode) && (mode&ENABLE_LINE_INPUT)) {
        std::wstring wide;
        for (;;) {
            wchar_t chunk[512];
            DWORD count=0;
            if (!ReadConsoleW(handle,chunk,512,&count,nullptr) || !count) {
                line.clear();
                return ctx.is_interrupted.exchange(false);
            }
            wide.append(chunk,count);
            if (wide.back()==L'\n') break;
        }
        wide.pop_back();
        if (!wide.empty() && wide.back()==L'\r') wide.pop_back();
        if (wide==L"\x1a") return false;
        line.clear();
        if (wide.empty()) return true;
        const int n=wide.size()<=static_cast<size_t>(INT_MAX) ? static_cast<int>(wide.size()) : 0;
        const int needed=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide.data(),n,nullptr,0,nullptr,nullptr);
        if (needed<=0) { ShellPrint("[Input] Invalid Unicode console line.\n"); return true; }
        line.resize(static_cast<size_t>(needed));
        if (!WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide.data(),n,line.data(),needed,nullptr,nullptr)) {
            line.clear(); ShellPrint("[Input] Could not read UTF-8 text.\n");
        }
        return true;
    }
#endif
    return bool(std::getline(std::cin,line));
}

//Shell文本打印走这里
void MiciusShell::ShellPrint(const std::string& msg)
{
    std::cout << msg;
    if (ctx.is_logging) LogText(StripANSI(msg),msg.find('\n')!=std::string::npos);
}

std::vector<std::string> MiciusShell::TokenizeInput(const std::string& input)
{
    std::vector<std::string> tokens;
    std::string token;
    bool quoted=false, started=false;
    for (char c:input) {
        if (c=='"') { quoted=!quoted; started=true; }
        else if (!quoted && (c==' ' || c=='\t' || c=='\r')) {
            if (started) { tokens.push_back(token); token.clear(); started=false; }
        } else { token+=c; started=true; }
    }
    if (quoted) throw std::invalid_argument("Unclosed double quote in command");
    if (started) tokens.push_back(token);
    return tokens;
}

void MiciusShell::RunREPL()
{
    std::string input_line;
    while (!ctx.should_exit) {
        std::cout << "\033[1;36m" << ctx.prompt_text << "\033[0m" << std::flush;
        
        if (!ReadLine(input_line)) break;
        if (input_line.empty()) continue;
        
        //如果开启了日志，原样存档用户输入的命令（清除提示符的颜色代码）
        LogInput(ctx.prompt_text,input_line);

        std::vector<std::string> args;
        try { args=TokenizeInput(input_line); }
        catch (const std::invalid_argument& e) { ShellPrint(std::string("[Error] ")+e.what()+"\n"); continue; }
        if (args.empty()) continue;
        
        const std::string& cmd_name = args[0];
        
        //从哈希表中提取函数指针
        auto it = cmd_router.find(cmd_name);
        if (it != cmd_router.end()) {
            try {
                it->second.handler(this, &ctx, args);
            } catch (const std::exception& error) {
                ShellPrint(
                    std::string("\033[1;31m[Error]\033[0m Command aborted by exception: ") +
                    error.what() + "\n");
            } catch (...) {
                ShellPrint("\033[1;31m[Error]\033[0m Command aborted by an unknown exception.\n");
            }
        } else {
            ShellPrint(GetMsg("msg_unknown_cmd"));
        }
    }
}

int MiciusShell::RunModelBenchmark(const std::vector<std::string>& args)
{
#if defined(MICIUS_I6_BENCHMARK)
    if (!args.empty() && args.front()=="--i6-mode") return RunI6Benchmark(args);
#endif
    struct BenchmarkOptions {
        std::string model_path;
        std::string pw2_parity_path;
#if defined(MICIUS_I4_EVALUATION)
        std::string i4_tokens, i4_output;
#endif
        uint64_t warmup_steps = 0;
        uint64_t timed_steps = 0;
        uint32_t cpu_id = 0;
        bool cpu_explicit = false;
        bool steps_explicit = false;
        bool itt_roi = false;
        bool d1_profile = false;
        std::vector<uint32_t> prompt_tokens;
    } options;

    const auto print_usage = []() {
        std::cout
            << "Usage: micius_cli --model-benchmark --model PATH --warmup N "
               "--steps N [--cpu N] [--itt-roi] [--d1-profile] [--pw2-parity PATH] --token ID [--token ID ...]\n";
    };
    const auto fail = [&](const std::string& message) {
        std::cout << "status=FAIL\nerror=" << message << '\n';
        return 1;
    };
    const auto require_value = [&](size_t* index, const char* option) -> const std::string* {
        if (!index || *index + 1U >= args.size()) {
            std::cout << "status=FAIL\nerror=" << option << " requires a value\n";
            return nullptr;
        }
        ++(*index);
        return &args[*index];
    };

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& argument = args[i];
        if (argument == "--help") {
            print_usage();
            return 0;
        }
#if defined(MICIUS_I4_EVALUATION)
        if (argument == "--i4-tokens" || argument == "--i4-output") {
            const std::string* value=require_value(&i,argument.c_str());
            if (!value || value->empty()) return fail("I4 options require nonempty paths");
            (argument=="--i4-tokens" ? options.i4_tokens : options.i4_output)=*value;
            continue;
        }
#endif
        if (argument == "--model") {
            const std::string* value = require_value(&i, "--model");
            if (!value) return 2;
            options.model_path = *value;
        } else if (argument == "--pw2-parity") {
            const std::string* value = require_value(&i, "--pw2-parity");
            if (!value || value->empty()) return fail("--pw2-parity requires a path");
            options.pw2_parity_path = *value;
        } else if (argument == "--d1-profile") {
            options.d1_profile = true;
        } else if (argument == "--itt-roi") {
            options.itt_roi = true;
        } else if (argument == "--warmup") {
            const std::string* value = require_value(&i, "--warmup");
            if (!value) return 2;
            if (!ParseBenchmarkU64(*value, &options.warmup_steps)) {
                return fail("--warmup requires an unsigned integer");
            }
        } else if (argument == "--steps") {
            const std::string* value = require_value(&i, "--steps");
            if (!value) return 2;
            if (!ParseBenchmarkU64(*value, &options.timed_steps) || options.timed_steps == 0) {
                return fail("--steps requires a positive unsigned integer");
            }
            options.steps_explicit = true;
        } else if (argument == "--cpu") {
            const std::string* value = require_value(&i, "--cpu");
            if (!value) return 2;
            uint64_t parsed = 0;
            if (!ParseBenchmarkU64(*value, &parsed) ||
                parsed > std::numeric_limits<uint32_t>::max()) {
                return fail("--cpu requires a uint32_t value");
            }
            options.cpu_id = static_cast<uint32_t>(parsed);
            options.cpu_explicit = true;
        } else if (argument == "--token") {
            const std::string* value = require_value(&i, "--token");
            if (!value) return 2;
            uint64_t parsed = 0;
            if (!ParseBenchmarkU64(*value, &parsed) ||
                parsed > std::numeric_limits<uint32_t>::max()) {
                return fail("--token requires a uint32_t value");
            }
            options.prompt_tokens.push_back(static_cast<uint32_t>(parsed));
        } else {
            print_usage();
            return fail("unknown model-benchmark option: " + argument);
        }
    }

    if (options.model_path.empty()) return fail("--model is required");
#if defined(MICIUS_I4_EVALUATION)
    const bool i4_requested=!options.i4_tokens.empty() || !options.i4_output.empty();
    if (i4_requested) {
        if (options.i4_tokens.empty() || options.i4_output.empty() || options.steps_explicit ||
            !options.prompt_tokens.empty() || options.warmup_steps!=0 || options.itt_roi ||
            options.d1_profile || !options.pw2_parity_path.empty())
            return fail("I4 requires tokens/output together and no sampling, timing or other diagnostic options");
    } else
#endif
    {
    if (!options.steps_explicit) return fail("--steps is required");
    if (options.prompt_tokens.empty()) return fail("at least one --token is required");
    }
    if (!options.pw2_parity_path.empty() && (options.prompt_tokens.size()!=1 ||
        options.warmup_steps!=0 || options.itt_roi))
        return fail("PW2 parity uses one repeated fixed token, warmup=0, ITT disabled");

    if (options.d1_profile && !micius::d1::compiled)
        return fail("D1 profile is not compiled in this executable");
    if (options.d1_profile && options.itt_roi)
        return fail("D1 and ITT must not be combined");
    std::cout << "d1_profile_compiled=" << micius::d1::compiled
              << "\nd1_profile_requested=" << options.d1_profile << '\n';
    micius::benchmark::IttRoi itt_roi;
    if (const char* error = itt_roi.Prepare(options.itt_roi)) return fail(error);

    // Pin before loading too: preparation timing and first-touch use the same CPU.
    if (options.cpu_explicit) {
        int load_cpu = -1; std::string error;
        if (!PinBenchmarkThread(options.cpu_id, &load_cpu, &error)) return fail(error);
    }
    const auto load_begin = std::chrono::steady_clock::now();
    if (!LoadFile(options.model_path)) return fail("model load or V1 tensor binding failed");
#if defined(MICIUS_I4_EVALUATION)
    if (i4_requested) {
        const int result=micius::i4::EvaluateTokens(ctx.engine_ctx,ctx.engine_config,options.i4_tokens,options.i4_output);
        if (options.cpu_explicit && CurrentProcessorForBenchmark()!=static_cast<int>(options.cpu_id))
            return fail("I4 evaluation left requested CPU");
        return result;
    }
#endif
    const double model_load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_begin).count();
    const auto weight_stats = micius::internal::GetPreparedWeightStats(ctx.engine_ctx);
    std::cout << std::setprecision(12)
        << "pw2_revision=pw2-2026-09-20\n"
        << "pw2_selection=" << weight_stats.selection << '\n'
        << "pw2_status=" << weight_stats.status << '\n'
        << "pw2_matrices=" << weight_stats.matrices << '\n'
        << "pw2_budget_bytes=" << weight_stats.budget_bytes << '\n'
        << "pw2_extra_bytes=" << weight_stats.extra_bytes << '\n'
        << "pw2_preparation_seconds=" << weight_stats.preparation_seconds << '\n'
        << "model_load_and_bind_seconds=" << model_load_seconds << '\n';

    for (uint32_t token : options.prompt_tokens) {
        if (token >= ctx.engine_config.vocab_size) {
            return fail("prompt token is outside the model vocabulary");
        }
    }

    const uint64_t prompt_count = static_cast<uint64_t>(options.prompt_tokens.size());
    if (options.warmup_steps > std::numeric_limits<uint64_t>::max() - prompt_count ||
        options.timed_steps >
            std::numeric_limits<uint64_t>::max() - prompt_count - options.warmup_steps) {
        return fail("prompt, warmup and timed position count overflows uint64_t");
    }
    const uint64_t total_positions =
        prompt_count + options.warmup_steps + options.timed_steps;
    if (total_positions > ctx.engine_config.max_seq_len) {
        return fail("prompt, warmup and timed positions exceed model context length");
    }
    if (options.timed_steps > std::numeric_limits<size_t>::max()) {
        return fail("--steps exceeds the host address space");
    }

    int observed_cpu = CurrentProcessorForBenchmark();
    if (options.cpu_explicit) {
        std::string affinity_error;
        if (!PinBenchmarkThread(options.cpu_id, &observed_cpu, &affinity_error)) {
            return fail(affinity_error);
        }
    }

    if (!Micius_ResetEngineV1(ctx.engine_ctx)) {
        return fail(BenchmarkApiError("Micius_ResetEngineV1"));
    }

    if (!options.pw2_parity_path.empty()) {
        // Separate correctness run: full logits, fixed inputs, no throughput sample.
        std::ofstream trace(options.pw2_parity_path, std::ios::binary | std::ios::trunc);
        if (!trace) return fail("cannot create PW2 parity trace");
        std::vector<float> logits(ctx.engine_config.vocab_size);
        const uint32_t vocabulary = ctx.engine_config.vocab_size;
        trace.write(reinterpret_cast<const char*>(&vocabulary), sizeof(vocabulary));
        const uint32_t token = options.prompt_tokens.front();
        micius::d1::Session d1_session(options.d1_profile);
        if (!d1_session.valid()) return fail("D1 session already active");
        for (uint64_t step=0; step<options.timed_steps; ++step) {
            uint32_t output=0;
            if (Micius_DecodeOneV1(ctx.engine_ctx, token, &output)!=MICIUS_STATE_YIELD_TOKEN ||
                !Micius_CopyLastLogitsV1(ctx.engine_ctx, logits.data(), logits.size()))
                return fail(BenchmarkApiError("PW2 fixed-input decode/logits"));
            trace.write(reinterpret_cast<const char*>(&token), sizeof(token));
            trace.write(reinterpret_cast<const char*>(&output), sizeof(output));
            trace.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size()*sizeof(float)));
            if (!trace) return fail("PW2 trace write failed");
        }
        if (!d1_session.stop()) return fail("D1 scope did not finish");
        if (options.d1_profile) micius::d1::Emit(std::cout, "parity", 0,
            ctx.engine_config.num_layers, options.timed_steps);
        trace.close();
        if (!trace) return fail("PW2 trace close failed");
        const int end_cpu=CurrentProcessorForBenchmark();
        if (options.cpu_explicit && end_cpu!=static_cast<int>(options.cpu_id))
            return fail("PW2 parity left requested CPU");
        std::cout << "mode=pw2_fixed_token_full_logits\n"
            << "pw2_trace_steps=" << options.timed_steps << '\n'
            << "pw2_trace_token=" << token << '\n'
            << "worker_start_cpu_0=" << observed_cpu << '\n'
            << "worker_end_cpu_0=" << end_cpu << "\nstatus=PASS\n";
        return 0;
    }

    std::vector<uint32_t> timed_tokens(static_cast<size_t>(options.timed_steps));
    std::vector<float> last_logits(ctx.engine_config.vocab_size);
    uint32_t current_token = 0;
    for (size_t i = 0; i < options.prompt_tokens.size(); ++i) {
        const bool last = i + 1U == options.prompt_tokens.size();
        const MiciusStatus status = last
            ? Micius_DecodeOneV1(ctx.engine_ctx, options.prompt_tokens[i], &current_token)
            : Micius_PrefillTokenV1(ctx.engine_ctx, options.prompt_tokens[i]);
        const MiciusStatus expected =
            last ? MICIUS_STATE_YIELD_TOKEN : MICIUS_STATE_TOKEN_ACCEPTED;
        if (status != expected) {
            return fail(BenchmarkApiError(last
                ? "prompt Micius_DecodeOneV1"
                : "Micius_PrefillTokenV1"));
        }
    }

    for (uint64_t step = 0; step < options.warmup_steps; ++step) {
        uint32_t output_token = 0;
        if (Micius_DecodeOneV1(ctx.engine_ctx, current_token, &output_token) !=
            MICIUS_STATE_YIELD_TOKEN) {
            return fail(BenchmarkApiError("warmup Micius_DecodeOneV1"));
        }
        current_token = output_token;
    }

    const int start_cpu = CurrentProcessorForBenchmark();
    if (options.cpu_explicit &&
        (start_cpu < 0 || static_cast<uint32_t>(start_cpu) != options.cpu_id)) {
        return fail("benchmark thread left the requested CPU before the timed ROI");
    }

    std::cout
        << "MICIUS real-model reference benchmark\n"
        << "abi=0x" << std::hex << MICIUS_ABI_VERSION << std::dec << '\n'
        << "gemv_backend=" << micius::backend::GEMVBackend(ctx.engine_ctx) << '\n'
#if defined(MICIUS_REFERENCE_GEMV_AVX2)
        << "gemv_fp_contract=off\n"
#else
        << "gemv_fp_contract=compiler_default_scalar_reference\n"
#endif
        << "gemv_input_preparation=" << micius::backend::InputPreparation(ctx.engine_ctx) << '\n'
        << "gemv_numerical_contract=" << micius::backend::NumericalContract(ctx.engine_ctx) << '\n'
        << "gemv_scale_validation=" << micius::backend::ScaleValidation(ctx.engine_ctx) << '\n'
        << "gemv_scale_storage=" << micius::backend::ScaleStorage(ctx.engine_ctx) << '\n'
        << "attention_value_backend=" << micius::backend::AttentionValue() << '\n'
        << "attention_score_query_cache=" << micius::backend::QueryCache() << '\n'
        << "attention_score_time_tile=" << micius::backend::ScoreTile() << '\n'
        << "mode=real_model_decode_throughput\n"
        << "profiling_roi=" << (options.itt_roi ? "itt_decode_only" : "disabled") << '\n'
        << "itt_control_callbacks=" << (options.itt_roi ? "bound" : "not_requested") << '\n'
        << "prompt_tokens=" << options.prompt_tokens.size() << '\n'
        << "warmup_steps=" << options.warmup_steps << '\n'
        << "timed_steps=" << options.timed_steps << '\n'
        << "cpu_affinity="
        << (options.cpu_explicit ? static_cast<int64_t>(options.cpu_id) : int64_t{-1})
        << '\n'
        << "VTUNE_ROI_BEGIN\n" << std::flush;

    micius::d1::Session d1_session(options.d1_profile);
    if (!d1_session.valid()) return fail("D1 session already active");
    itt_roi.Begin();
    const auto begin = std::chrono::steady_clock::now();
    uint64_t completed_steps = 0;
    for (; completed_steps < options.timed_steps; ++completed_steps) {
        uint32_t output_token = 0;
        if (Micius_DecodeOneV1(ctx.engine_ctx, current_token, &output_token) !=
            MICIUS_STATE_YIELD_TOKEN) {
            break;
        }
        timed_tokens[static_cast<size_t>(completed_steps)] = output_token;
        current_token = output_token;
    }
    const auto end = std::chrono::steady_clock::now();
    itt_roi.End();
    if (!d1_session.stop()) return fail("D1 scope did not finish");
    const int end_cpu = CurrentProcessorForBenchmark();
    std::cout << "VTUNE_ROI_END\n";

    if (completed_steps != options.timed_steps) {
        return fail(BenchmarkApiError("timed Micius_DecodeOneV1"));
    }
    if (options.cpu_explicit &&
        (end_cpu < 0 || static_cast<uint32_t>(end_cpu) != options.cpu_id)) {
        return fail("benchmark thread left the requested CPU during the timed ROI");
    }
    if (!Micius_CopyLastLogitsV1(
            ctx.engine_ctx,
            last_logits.data(),
            last_logits.size())) {
        return fail(BenchmarkApiError("Micius_CopyLastLogitsV1"));
    }

    constexpr uint64_t kHashBasis = UINT64_C(14695981039346656037);
    uint64_t token_hash = kHashBasis;
    for (uint32_t token : timed_tokens) token_hash = HashBenchmarkWord(token_hash, token);
    uint64_t logits_hash = kHashBasis;
    for (float logit : last_logits) {
        logits_hash = HashBenchmarkWord(logits_hash, std::bit_cast<uint32_t>(logit));
    }

    const double wall_seconds = std::chrono::duration<double>(end - begin).count();
    const double steps_per_second =
        wall_seconds > 0.0
            ? static_cast<double>(completed_steps) / wall_seconds
            : 0.0;
    std::cout
        << std::fixed << std::setprecision(6)
        << "status=PASS\n"
        << "wall_seconds=" << wall_seconds << '\n'
        << "completed_steps=" << completed_steps << '\n'
        << "steps_per_second=" << steps_per_second << '\n'
        << "worker_start_cpu_0=" << start_cpu << '\n'
        << "worker_end_cpu_0=" << end_cpu << '\n'
        << "final_token=" << current_token << '\n'
        << "token_checksum=0x" << std::hex << token_hash << '\n'
        << "last_logits_checksum=0x" << logits_hash << std::dec << '\n';
    if (options.d1_profile) micius::d1::Emit(std::cout, "decode",
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()),
        ctx.engine_config.num_layers, completed_steps);
    return 0;
}

//============================================================================
//命令回调区
//============================================================================

void MiciusShell::CmdHelp(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    (void)ctx;
    (void)args;
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
    (void)ctx;
    (void)args;
    //发送清屏并复位光标
    shell->ShellPrint("\033[2J\033[1;1H");
}

void MiciusShell::CmdLog(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (args.size() < 2) { shell->ShellPrint(shell->GetMsg("err_usage_log")); return; }
    
    if (args[1].compare("start") == 0) {
        if (args.size() != 3 || args[2].empty()) { shell->ShellPrint(shell->GetMsg("err_usage_log")); return; }
        if (ctx->is_logging) { shell->ShellPrint("Logging is already active.\n"); return; }
        
        ctx->log_file.clear();
        ctx->log_file.open(micius::gguf::NativePath(std::filesystem::u8path(args[2])), std::ios::app);
        if (!ctx->log_file.is_open()) {
            shell->ShellPrint("\033[1;31m[Error]\033[0m Failed to open log file.\n");
        } else {
            ctx->is_logging = true;
            std::string msg;
            msg.assign("[System] Archiving started to: ").append(args[2]).append("\n");
            shell->ShellPrint(msg);
        }
    } 
    else if (args[1].compare("stop") == 0 && args.size()==2) {
        if (!ctx->is_logging) { shell->ShellPrint("Logging is not active.\n"); return; }
        ctx->log_file.close();
        ctx->is_logging = false;
        shell->ShellPrint(ctx->log_file.fail() ? "[Log] Session log close failed.\n" : "[System] Archiving stopped.\n");
    } 
    else { shell->ShellPrint(shell->GetMsg("err_usage_log")); }
}

void MiciusShell::CmdExit(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    (void)args;
    shell->ShellPrint(shell->GetMsg("msg_exit"));
    ctx->should_exit = true;
}

bool MiciusShell::LoadFile(const std::string& filename, uint32_t context_length)
{
    namespace fs = std::filesystem;
    auto path = fs::u8path(ctx.model_dir) / fs::u8path(filename);
    if (!fs::exists(path)) path = fs::u8path(filename);
    const auto utf8 = path.u8string();
    const std::string utf8_path(utf8.begin(),utf8.end());
    const MiciusGGUFParserV1* parsers[] = {
#if defined(MICIUS_LINK_LLAMA)
        Micius_LlamaParserV1(),
#endif
#if defined(MICIUS_LINK_QWEN3)
        Micius_Qwen3ParserV1(),
#endif
#if defined(MICIUS_LINK_QWEN2)
        Micius_Qwen2ParserV1(),
#endif
    };
    const char* cache_root = std::getenv("MICIUS_CACHE_ROOT");
    MiciusGGUFOpenOptionsV1 o{MICIUS_GGUF_FILE_ABI,sizeof(o),utf8_path.c_str(),
        cache_root ? cache_root : ".micius-cache",parsers,static_cast<uint32_t>(std::size(parsers)),context_length};
    char diagnostic[512]{};
    std::unique_ptr<MiciusFileModel, decltype(&Micius_CloseGGUFV1)> next(
        Micius_OpenGGUFV1(&o,diagnostic,sizeof(diagnostic)),Micius_CloseGGUFV1);
    if (!next) {
        ShellPrint(std::string("[Error] Model load failed: ") + diagnostic + "\n");
        return false;
    }
    const auto stem = path.stem().u8string();
    std::string name(stem.begin(),stem.end());
    std::string prompt = name + "> ";
    const std::string message = "[Loader] " + name + " securely bound to Engine.\narchitecture=" +
        Micius_FileArchitectureV1(next.get()) + "\ncontext_directory=" + Micius_FileContextDirectoryV1(next.get()) + "\n";
    ReleaseFile(&ctx);
    ctx.file_model = next.release();
    ctx.engine_ctx = Micius_FileEngineV1(ctx.file_model);
    ctx.engine_config = *Micius_FileConfigV1(ctx.file_model);
    ctx.active_model.swap(name);
    ctx.prompt_text.swap(prompt);
    ShellPrint(message);
    return true;
}

void MiciusShell::CmdLoad(MiciusShell* shell, ShellContext*, const std::vector<std::string>& args)
{
    if (args.size() != 2) { shell->ShellPrint(shell->GetMsg("err_usage_load")); return; }
    shell->LoadFile(args[1]);
}

void MiciusShell::CmdGenerate(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (!ctx->engine_ctx) {
        shell->ShellPrint(shell->GetMsg("err_no_engine"));
        return;
    }
    
    // This checkpoint intentionally accepts only explicit token IDs. The previous
    // whitespace/hash fallback was not a tokenizer and produced false language.
    std::vector<uint32_t> prompt_tokens;
    std::stringstream prompt_summary;

    for (size_t i = 1; i < args.size(); ++i) {
        try {
            size_t parsed_chars = 0;
            unsigned long value = std::stoul(args[i], &parsed_chars);
            if (parsed_chars != args[i].size() || value >= ctx->engine_config.vocab_size) {
                throw std::out_of_range("token ID outside vocabulary");
            }
            prompt_tokens.push_back(static_cast<uint32_t>(value));
            prompt_summary << " [" << value << "]";
        } catch (...) {
            shell->ShellPrint(
                "\033[1;31m[Error]\033[0m Text tokenization is not implemented in this checkpoint. "
                "Use explicit numeric token IDs.\n");
            return;
        }
    }

    if (prompt_tokens.empty()) {
        shell->ShellPrint("Usage: generate <token_id> [token_id ...]\n");
        return;
    }

    ctx->is_interrupted.store(false, std::memory_order_relaxed);
    shell->ShellPrint("\033[1;33m[System] Prompt Ingestion:\033[0m" + prompt_summary.str() + "\n");
    
    if (!Micius_ResetEngineV1(ctx->engine_ctx)) {
        shell->ShellPrint(
            std::string("\033[1;31m[Error]\033[0m Engine reset rejected: ") +
            Micius_GetLastError() + "\n");
        return;
    }
    
    uint32_t out_token = 0;
    MiciusStatus status;

    //Prefill阶段
    //串行输入所有的Prompt Token以累积KV Cache
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        uint32_t tk = prompt_tokens[i];

        const bool is_last_prompt_token = i + 1 == prompt_tokens.size();
        status = is_last_prompt_token
            ? Micius_DecodeOneV1(ctx->engine_ctx, tk, &out_token)
            : Micius_PrefillTokenV1(ctx->engine_ctx, tk);
        const MiciusStatus expected_status = is_last_prompt_token
            ? MICIUS_STATE_YIELD_TOKEN
            : MICIUS_STATE_TOKEN_ACCEPTED;
        if (status != expected_status) {
            shell->ShellPrint(
                std::string("\033[1;31m[Error]\033[0m Prefill stopped: ") +
                Micius_GetLastError() + "\n");
            return;
        }
    }

    shell->ShellPrint("\033[1;32m[System] Prefill complete. Executing reference decode...\033[0m\nAssistant: ");

    //Decode阶段
    //基于最后输出的Token进行连贯的自回归生成
    uint32_t current_token = out_token;
    uint32_t token_count = 0;

    while (!ctx->is_interrupted.load(std::memory_order_relaxed)) {
        // The final prompt decode has already produced current_token. Emit it
        // before feeding it back; decoding first silently dropped token T1 and
        // began the visible stream at T2.
        shell->ShellPrint(" \033[1;32m[" + std::to_string(current_token) + "]\033[0m");
        ++token_count;

        if (token_count >= 150) {
            shell->ShellPrint("\n\033[1;33m[System] Safe boundary hit (150 tokens).\033[0m\n");
            break;
        }

        status = Micius_DecodeOneV1(ctx->engine_ctx, current_token, &out_token);
        if (status != MICIUS_STATE_YIELD_TOKEN) {
            if (status == MICIUS_STATE_TREE_REJECTED) {
                shell->ShellPrint(
                    "\n\033[1;31m[Error]\033[0m Tree rejection requires a real target-model fallback token.\n");
            } else {
                shell->ShellPrint(
                    std::string("\n\033[1;31m[Error]\033[0m Decode stopped: ") +
                    Micius_GetLastError() + "\n");
            }
            // CONTEXT_FULL, NUMERIC_ERROR, OOM, NOT_READY and invalid input are
            // all terminal for this synchronous generation command.
            break;
        }

        current_token = out_token;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    if (ctx->is_interrupted.load(std::memory_order_relaxed)) {
        shell->ShellPrint("\n\033[1;31m[System] Output halted by user (SIGINT).\033[0m\n");
        std::cin.clear(); 
    } else {
        shell->ShellPrint("\n");
    }

}

void MiciusShell::CmdStats(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    (void)args;
    if (!ctx->engine_ctx) { shell->ShellPrint(shell->GetMsg("err_no_engine")); return; }
    
    MiciusRuntimeStats stats;
    Micius_GetRuntimeStats(ctx->engine_ctx, &stats);
    uint32_t worker_count = Micius_GetWorkerCount(ctx->engine_ctx);
    
    const double kv_usage = stats.kv_slots_total > 0
        ? static_cast<double>(stats.kv_slots_used) /
          static_cast<double>(stats.kv_slots_total) * 100.0
        : 0.0;

    std::stringstream ss;
    ss << "\n========== MICIUS Runtime Telemetry ==========\n";
    ss << " Tokens Generated : " << stats.total_tokens_generated << "\n";
    ss << " SpecTree Rejects : " << stats.tree_reject_count << " (Hostile Churn)\n";
    ss << " KV Pool Usage    : [";
    constexpr int bar_width = 20;
    const int filled = static_cast<int>(kv_usage / 100.0 * static_cast<double>(bar_width));
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
