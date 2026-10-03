#include "MiciusShell.h"
#include "MiciusTensorCodec.h"
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

using ShellQuantDescriptor = MiciusQ8Block32Scale;
static_assert(sizeof(ShellQuantDescriptor) == 16);

static bool CheckedProduct(size_t lhs, size_t rhs, size_t* result) noexcept
{
    if (!result || (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static bool GetTensorElementCount(const GGUFTensorInfo& tensor, size_t* count) noexcept
{
    if (!count || tensor.n_dims == 0 || tensor.n_dims > 4) return false;
    size_t product = 1;
    for (uint32_t i = 0; i < tensor.n_dims; ++i) {
        if (tensor.dimensions[i] == 0 ||
            tensor.dimensions[i] > std::numeric_limits<size_t>::max() ||
            !CheckedProduct(product, static_cast<size_t>(tensor.dimensions[i]), &product)) {
            return false;
        }
    }
    *count = product;
    return true;
}

static MiciusTensorEntryV1 MakeBridgeEntry(
    const char* name,
    const void* data,
    size_t byte_size,
    MiciusTensorDataType type,
    uint32_t rank,
    uint64_t rows,
    uint64_t columns,
    size_t row_stride_bytes) noexcept
{
    MiciusTensorEntryV1 entry{};
    entry.tensor_name = name;
    entry.tensor_data = data;
    entry.byte_size = byte_size;
    entry.data_type = static_cast<uint32_t>(type);
    entry.rank = rank;
    entry.dimensions[0] = rows;
    entry.dimensions[1] = columns;
    entry.row_stride_bytes = row_stride_bytes;
    return entry;
}

static const char* BenchmarkGEMVBackendName() noexcept
{
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    return "q8_32_prepared_f64_avx2_row_pair_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
    return "q8_32_f32_avx2_row_pair_direct_load_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_AVX2)
    return "q8_32_f32_avx2_experimental";
#else
    return "scalar_double_reference";
#endif
}

static const char* BenchmarkScaleValidationName() noexcept
{
#if defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    return "private_v1_prevalidated_checked_fallback";
#else
    return "per_call_checked";
#endif
}

static const char* BenchmarkScaleStorageName() noexcept
{
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    return "private_f32_per_block_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    return "private_q8_32_descriptor_per_block";
#else
    return "q8_32_descriptor_checked";
#endif
}

static const char* BenchmarkAttentionValueName() noexcept
{
#if defined(MICIUS_REFERENCE_ATTENTION_VALUE_AVX2)
    return "f32_f64_avx2_order_preserving_experimental";
#else
    return "scalar_f64_reference";
#endif
}

static const char* BenchmarkAttentionQueryCacheName() noexcept
{
#if defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
    return "per_head_exact_f64_experimental";
#elif defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
    return "none_time_quad_shared_conversion";
#else
    return "per_dot_f32_to_f64";
#endif
}

static const char* BenchmarkAttentionTimeTileName() noexcept
{
#if defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
    return "four_rows_shared_query_experimental";
#else
    return "single_row";
#endif
}

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
    ctx.model_parser = nullptr;
}

MiciusShell::~MiciusShell()
{
    if (ctx.is_logging && ctx.log_file.is_open()) {
        ctx.log_file.close();
    }
    
    if (ctx.engine_ctx) {
        Micius_DestroyEngine(ctx.engine_ctx);
        ctx.engine_ctx = nullptr;
    }

    ClearManagedBuffers(&ctx);

    if (ctx.model_parser) {
        delete ctx.model_parser;
        ctx.model_parser = nullptr;
    }

    if (g_active_shell == this) g_active_shell = nullptr;
}

void MiciusShell::ClearManagedBuffers(ShellContext* ctx)
{
    for (void* ptr : ctx->managed_buffers) {
        if (ptr) {
            _mm_free(ptr);
        }
    }
    ctx->managed_buffers.clear();
    ctx->dynamic_tensor_names.clear();
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
    cmd_router.insert({"quit", {"Terminate the runtime shell (alias)", CmdExit}});
    cmd_router.insert({"load", {"Load a GGUF model", CmdLoad}});
    cmd_router.insert({"stats", {"Show telemetry and hardware bounds", CmdStats}});
    cmd_router.insert({"debug", {"Debug underlying mechanics (e.g. debug workers)", CmdDebugWorkers}});
    cmd_router.insert({"clear", {"Clear the console screen", CmdClear}});
    cmd_router.insert({"cls", {"Clear the console screen (alias)", CmdClear}});
    cmd_router.insert({"log", {"Archive session (usage: log start <file> | log stop)", CmdLog}});
    cmd_router.insert({"generate", {"Start model generation loop (Ctrl+C to interrupt)", CmdGenerate}});
    cmd_router.insert({"run", {"Alias for generate", CmdGenerate}});

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
            try {
                it->second.handler(this, &ctx, args);
            } catch (const std::exception& error) {
                ShellPrint(
                    std::string("\033[1;31m[Error]\033[0m Command aborted by exception: ") +
                    error.what() + "\n");
                if (ctx.engine_ctx) {
                    Micius_DestroyEngine(ctx.engine_ctx);
                    ctx.engine_ctx = nullptr;
                }
                ClearManagedBuffers(&ctx);
                delete ctx.model_parser;
                ctx.model_parser = nullptr;
                ctx.active_model.clear();
                ctx.prompt_text.assign("micius> ");
            } catch (...) {
                ShellPrint("\033[1;31m[Error]\033[0m Command aborted by an unknown exception.\n");
                if (ctx.engine_ctx) {
                    Micius_DestroyEngine(ctx.engine_ctx);
                    ctx.engine_ctx = nullptr;
                }
                ClearManagedBuffers(&ctx);
                delete ctx.model_parser;
                ctx.model_parser = nullptr;
                ctx.active_model.clear();
                ctx.prompt_text.assign("micius> ");
            }
        } else {
            ShellPrint(GetMsg("msg_unknown_cmd"));
        }
    }
}

int MiciusShell::RunModelBenchmark(const std::vector<std::string>& args)
{
    struct BenchmarkOptions {
        std::string model_path;
        std::string pw2_parity_path;
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
    if (!options.steps_explicit) return fail("--steps is required");
    if (options.prompt_tokens.empty()) return fail("at least one --token is required");
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
    CmdLoad(this, &ctx, std::vector<std::string>{"load", options.model_path});
    if (!ctx.engine_ctx) return fail("model load or V1 tensor binding failed");
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
        << "gemv_backend=" << BenchmarkGEMVBackendName() << '\n'
#if defined(MICIUS_REFERENCE_GEMV_AVX2)
        << "gemv_fp_contract=off\n"
#else
        << "gemv_fp_contract=compiler_default_scalar_reference\n"
#endif
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
        << "gemv_input_preparation=f64_snapshot_per_input\n"
#else
        << "gemv_input_preparation=inline_f32_conversion\n"
#endif
        << "gemv_scale_validation=" << BenchmarkScaleValidationName() << '\n'
        << "gemv_scale_storage=" << BenchmarkScaleStorageName() << '\n'
        << "attention_value_backend=" << BenchmarkAttentionValueName() << '\n'
        << "attention_score_query_cache=" << BenchmarkAttentionQueryCacheName() << '\n'
        << "attention_score_time_tile=" << BenchmarkAttentionTimeTileName() << '\n'
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
    (void)args;
    shell->ShellPrint(shell->GetMsg("msg_exit"));
    ctx->should_exit = true;
}

void MiciusShell::CmdLoad(MiciusShell* shell, ShellContext* ctx, const std::vector<std::string>& args)
{
    if (args.size() < 2) {
        shell->ShellPrint(shell->GetMsg("err_usage_load"));
        return;
    }
    
    std::string file_name = args[1];
    std::filesystem::path target_path = std::filesystem::path(ctx->model_dir) / file_name;

    std::string msg_parse;
    msg_parse.assign("[Loader] Searching for ").append(target_path.string()).append("...\n");
    shell->ShellPrint(msg_parse);

    if (!std::filesystem::exists(target_path) && std::filesystem::exists(file_name)) {
        target_path = file_name; 
    }

    if (!std::filesystem::exists(target_path)) {
        shell->ShellPrint("\033[1;31m[Error]\033[0m Model file not found in models/ directory.\n");
        return;
    }

    //重建运行态前先清空旧内存
    if (ctx->engine_ctx) {
        Micius_DestroyEngine(ctx->engine_ctx);
        ctx->engine_ctx = nullptr;
    }
    if (ctx->model_parser) {
        delete ctx->model_parser;
        ctx->model_parser = nullptr;
    }
    ClearManagedBuffers(ctx);
    ctx->active_model.clear();
    ctx->prompt_text.assign("micius> ");
    ctx->engine_config = MiciusConfig{};

    GGUFParser* loader = nullptr;
    try {
        loader = new GGUFParser(target_path.string());
    } catch (const std::exception& e) {
        shell->ShellPrint(std::string("\033[1;31m[Error]\033[0m Failed to memory-map model: ") + e.what() + "\n");
        return;
    }

    try {
        if (!loader->ParseHeaders()) {
            shell->ShellPrint("\033[1;31m[Error]\033[0m Unsupported or invalid GGUF v2/v3 file.\n");
            delete loader;
            return;
        }
    } catch (const std::exception& error) {
        shell->ShellPrint(
            std::string("\033[1;31m[Error]\033[0m Malformed GGUF metadata: ") +
            error.what() + "\n");
        delete loader;
        return;
    }

    MiciusConfig config{};
    if (!Micius_InitConfig(&config, sizeof(config))) {
        shell->ShellPrint(
            std::string("\033[1;31m[Error]\033[0m Failed to initialize engine config: ") +
            Micius_GetLastError() + "\n");
        delete loader;
        return;
    }
    if (!loader->ExtractEngineConfig(config)) {
        shell->ShellPrint("\033[1;31m[Error]\033[0m Failed to extract configuration parameters.\n");
        delete loader;
        return;
    }

    loader->PrintTopology();
    ctx->engine_config = config;
    ctx->model_parser = loader; 
    ctx->engine_ctx = Micius_CreateEngineV1(
        &ctx->engine_config,
        sizeof(ctx->engine_config),
        MICIUS_ABI_VERSION);

    if (!ctx->engine_ctx) {
        shell->ShellPrint(
            std::string("\033[1;31m[Error]\033[0m Failed to instantiate Micius Engine: ") +
            Micius_GetLastError() + "\n");
        delete ctx->model_parser;
        ctx->model_parser = nullptr;
        return;
    }

    //解包GGUF并翻译为内核标准INT8权重格式
    std::vector<MiciusTensorEntryV1> bridge_entries;
    const auto& tensors = ctx->model_parser->GetTensors();
    if (tensors.size() > (std::numeric_limits<size_t>::max() - 1U) / 2U) {
        shell->ShellPrint("\033[1;31m[Error]\033[0m Tensor count exceeds host capacity.\n");
        Micius_DestroyEngine(ctx->engine_ctx);
        ctx->engine_ctx = nullptr;
        delete ctx->model_parser;
        ctx->model_parser = nullptr;
        return;
    }
    const size_t staging_capacity = tensors.size() * 2U + 1U;
    bridge_entries.reserve(staging_capacity);
    // At most two managed allocations are created per tensor, plus the F32
    // embedding view. Reserving first makes every later ownership handoff
    // non-throwing and prevents an untracked aligned allocation.
    ctx->managed_buffers.reserve(staging_capacity);
    ctx->dynamic_tensor_names.reserve(tensors.size() + 1U);

    bool translation_failed = false;
    std::string translation_error;

    for (const auto& t : tensors) {
        const void* raw_data = ctx->model_parser->GetTensorData(t.offset);
        if (!raw_data) {
            translation_failed = true;
            translation_error = "tensor offset is outside the GGUF data region for " + t.name;
            break;
        }
        
        if (t.name == "token_embd.weight") {
            size_t num_elements = 0;
            size_t fp32_bytes = 0;
            if (t.n_dims != 2 || !GetTensorElementCount(t, &num_elements) ||
                !CheckedProduct(num_elements, sizeof(float), &fp32_bytes)) {
                translation_failed = true;
                translation_error = "token embedding shape overflows the host address space";
                break;
            }
            
            //总是生成FP32版本供前向Embedding查表（改名为.f32）
            float* f32_embd = static_cast<float*>(_mm_malloc(fp32_bytes, 64));
            if (!f32_embd) {
                translation_failed = true;
                translation_error = "token embedding allocation failed";
                break;
            }
            if (t.type == GGMLType::F32) {
                std::memcpy(f32_embd, raw_data, fp32_bytes);
            } else if (t.type == GGMLType::F16) {
                const auto* f16_data = static_cast<const uint16_t*>(raw_data);
                bool finite_embedding = true;
                for (size_t i = 0; i < num_elements; ++i) {
                    f32_embd[i] = MiciusTensorCodec::DecodeFloat16(f16_data[i]);
                    finite_embedding = finite_embedding && std::isfinite(f32_embd[i]);
                }
                if (!finite_embedding) {
                    _mm_free(f32_embd);
                    translation_failed = true;
                    translation_error = "F16 token embedding contains a non-finite value";
                    break;
                }
            } else if (t.type == GGMLType::Q8_0) {
                if (t.dimensions[0] > std::numeric_limits<size_t>::max() ||
                    t.dimensions[1] > std::numeric_limits<size_t>::max() ||
                    !MiciusTensorCodec::DequantizeGGUFQ8_0ToF32(
                        raw_data,
                        static_cast<size_t>(t.byte_size),
                        static_cast<size_t>(t.dimensions[1]),
                        static_cast<size_t>(t.dimensions[0]),
                        f32_embd,
                        num_elements)) {
                    _mm_free(f32_embd);
                    translation_failed = true;
                    translation_error = "Q8_0 token embedding dequantization failed";
                    break;
                }
            } else {
                _mm_free(f32_embd);
                translation_failed = true;
                translation_error = "token embedding type is not supported by this checkpoint";
                break;
            }
            
            ctx->managed_buffers.push_back(f32_embd);
            const uint64_t embedding_columns = t.dimensions[0];
            const uint64_t embedding_rows = t.dimensions[1];
            bridge_entries.push_back(MakeBridgeEntry(
                "token_embd.weight.f32",
                f32_embd,
                fp32_bytes,
                MICIUS_TENSOR_F32,
                2,
                embedding_rows,
                embedding_columns,
                static_cast<size_t>(embedding_columns) * sizeof(float)));

            //对Q8_0生成INT8供可能的Tied LM Head复用
            if (t.type == GGMLType::Q8_0) {
                if (embedding_columns % 32U != 0 ||
                    embedding_rows > std::numeric_limits<size_t>::max() ||
                    embedding_columns > std::numeric_limits<size_t>::max()) {
                    translation_failed = true;
                    translation_error = "Q8_0 token embedding row shape is unsupported";
                    break;
                }
                size_t num_engine_blocks = 0;
                if (!CheckedProduct(
                        static_cast<size_t>(embedding_rows),
                        static_cast<size_t>(embedding_columns / 32U),
                        &num_engine_blocks)) {
                    translation_failed = true;
                    translation_error = "token embedding block count overflows";
                    break;
                }
                size_t scale_bytes = 0;
                if (!CheckedProduct(num_engine_blocks, sizeof(ShellQuantDescriptor), &scale_bytes)) {
                    translation_failed = true;
                    translation_error = "token embedding scale allocation size overflows";
                    break;
                }
                auto* w = static_cast<int8_t*>(_mm_malloc(num_elements, 64));
                auto* s = static_cast<ShellQuantDescriptor*>(_mm_malloc(scale_bytes, 64));
                if (!w || !s || !MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
                        raw_data,
                        static_cast<size_t>(t.byte_size),
                        static_cast<size_t>(embedding_rows),
                        static_cast<size_t>(embedding_columns),
                        w,
                        num_elements,
                        s,
                        num_engine_blocks)) {
                    if (w) _mm_free(w);
                    if (s) _mm_free(s);
                    translation_failed = true;
                    translation_error = "Q8_0 token embedding conversion failed";
                    break;
                }
                ctx->managed_buffers.push_back(w);
                ctx->managed_buffers.push_back(s);
                bridge_entries.push_back(MakeBridgeEntry(
                    t.name.c_str(),
                    w,
                    num_elements,
                    MICIUS_TENSOR_Q8_32_WEIGHT,
                    2,
                    embedding_rows,
                    embedding_columns,
                    static_cast<size_t>(embedding_columns)));
                ctx->dynamic_tensor_names.push_back("token_embd.scales");
                bridge_entries.push_back(MakeBridgeEntry(
                    ctx->dynamic_tensor_names.back().c_str(),
                    s,
                    scale_bytes,
                    MICIUS_TENSOR_Q8_32_SCALE,
                    2,
                    embedding_rows,
                    embedding_columns / 32U,
                    static_cast<size_t>(embedding_columns / 32U) * sizeof(ShellQuantDescriptor)));
            }
            continue; 
        }

        if (t.type == GGMLType::Q8_0) {
            size_t num_elements = 0;
            if (t.n_dims != 2 || !GetTensorElementCount(t, &num_elements) ||
                t.dimensions[0] % 32U != 0 ||
                t.dimensions[0] > std::numeric_limits<size_t>::max() ||
                t.dimensions[1] > std::numeric_limits<size_t>::max()) {
                translation_failed = true;
                translation_error = "Q8 tensor row shape is invalid for " + t.name;
                break;
            }

            const size_t columns = static_cast<size_t>(t.dimensions[0]);
            const size_t rows = static_cast<size_t>(t.dimensions[1]);
            size_t num_engine_blocks = 0;
            if (!CheckedProduct(rows, columns / 32U, &num_engine_blocks)) {
                translation_failed = true;
                translation_error = "Q8 block count overflows for " + t.name;
                break;
            }
            size_t scale_bytes = 0;
            if (!CheckedProduct(num_engine_blocks, sizeof(ShellQuantDescriptor), &scale_bytes)) {
                translation_failed = true;
                translation_error = "scale shape overflows for " + t.name;
                break;
            }
            auto* w = static_cast<int8_t*>(_mm_malloc(num_elements, 64));
            auto* s = static_cast<ShellQuantDescriptor*>(_mm_malloc(scale_bytes, 64));
            if (!w || !s || !MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
                    raw_data,
                    static_cast<size_t>(t.byte_size),
                    rows,
                    columns,
                    w,
                    num_elements,
                    s,
                    num_engine_blocks)) {
                if (w) _mm_free(w);
                if (s) _mm_free(s);
                translation_failed = true;
                translation_error = "Q8_0 conversion failed for " + t.name;
                break;
            }
            ctx->managed_buffers.push_back(w);
            ctx->managed_buffers.push_back(s);

            bridge_entries.push_back(MakeBridgeEntry(
                t.name.c_str(),
                w,
                num_elements,
                MICIUS_TENSOR_Q8_32_WEIGHT,
                2,
                rows,
                columns,
                columns));

            std::string weight_name = t.name;
            std::string scale_name;
            size_t last_dot = weight_name.find_last_of('.');
            if (last_dot != std::string::npos && weight_name.substr(last_dot) == ".weight") {
                scale_name = weight_name.substr(0, last_dot) + ".scales";
            } else {
                scale_name = weight_name + ".scales";
            }

            ctx->dynamic_tensor_names.push_back(scale_name);
            bridge_entries.push_back(MakeBridgeEntry(
                ctx->dynamic_tensor_names.back().c_str(),
                s,
                scale_bytes,
                MICIUS_TENSOR_Q8_32_SCALE,
                2,
                rows,
                columns / 32U,
                columns / 32U * sizeof(ShellQuantDescriptor)));
        } 
        else if (t.type == GGMLType::F32) {
            if (t.n_dims == 1) {
                bridge_entries.push_back(MakeBridgeEntry(
                    t.name.c_str(),
                    raw_data,
                    static_cast<size_t>(t.byte_size),
                    MICIUS_TENSOR_F32,
                    1,
                    t.dimensions[0],
                    0,
                    0));
            } else if (t.n_dims == 2 &&
                       t.dimensions[0] <= std::numeric_limits<size_t>::max()) {
                bridge_entries.push_back(MakeBridgeEntry(
                    t.name.c_str(),
                    raw_data,
                    static_cast<size_t>(t.byte_size),
                    MICIUS_TENSOR_F32,
                    2,
                    t.dimensions[1],
                    t.dimensions[0],
                    static_cast<size_t>(t.dimensions[0]) * sizeof(float)));
            } else {
                translation_failed = true;
                translation_error = "F32 tensor rank is unsupported for " + t.name;
                break;
            }
        } else if (t.type == GGMLType::F16 && t.n_dims == 1) {
            size_t num_elements = 0;
            size_t fp32_bytes = 0;
            if (!GetTensorElementCount(t, &num_elements) ||
                !CheckedProduct(num_elements, sizeof(float), &fp32_bytes)) {
                translation_failed = true;
                translation_error = "F16 vector shape overflows for " + t.name;
                break;
            }
            auto* converted = static_cast<float*>(_mm_malloc(fp32_bytes, 64));
            if (!converted) {
                translation_failed = true;
                translation_error = "F16 vector allocation failed for " + t.name;
                break;
            }
            const auto* source = static_cast<const uint16_t*>(raw_data);
            bool finite_vector = true;
            for (size_t i = 0; i < num_elements; ++i) {
                converted[i] = MiciusTensorCodec::DecodeFloat16(source[i]);
                finite_vector = finite_vector && std::isfinite(converted[i]);
            }
            if (!finite_vector) {
                _mm_free(converted);
                translation_failed = true;
                translation_error = "F16 vector contains a non-finite value for " + t.name;
                break;
            }
            ctx->managed_buffers.push_back(converted);
            bridge_entries.push_back(MakeBridgeEntry(
                t.name.c_str(),
                converted,
                fp32_bytes,
                MICIUS_TENSOR_F32,
                1,
                t.dimensions[0],
                0,
                0));
        } else {
            translation_failed = true;
            translation_error = "tensor type/rank is unsupported for " + t.name;
            break;
        }
    }

    if (translation_failed || bridge_entries.size() > std::numeric_limits<uint32_t>::max()) {
        shell->ShellPrint(
            std::string("\033[1;31m[Error]\033[0m Model translation aborted: ") +
            (translation_error.empty() ? "tensor entry count exceeds the ABI" : translation_error) +
            "\n");
        Micius_DestroyEngine(ctx->engine_ctx);
        ctx->engine_ctx = nullptr;
        ClearManagedBuffers(ctx);
        delete ctx->model_parser;
        ctx->model_parser = nullptr;
        return;
    }

    if (!Micius_BindModelTensorsV1(
            ctx->engine_ctx,
            bridge_entries.data(),
            static_cast<uint32_t>(bridge_entries.size()),
            sizeof(MiciusTensorEntryV1),
            MICIUS_ABI_VERSION)) {
        shell->ShellPrint(
            std::string("\033[1;31m[Error]\033[0m Tensor binding rejected: ") +
            Micius_GetLastError() + "\n");
        Micius_DestroyEngine(ctx->engine_ctx);
        ctx->engine_ctx = nullptr;
        ClearManagedBuffers(ctx);
        delete ctx->model_parser;
        ctx->model_parser = nullptr;
        return;
    }

    size_t last_dot = file_name.find_last_of('.');
    std::string model_name = (last_dot == std::string::npos) ? file_name : file_name.substr(0, last_dot);
    
    ctx->active_model.assign(model_name);
    ctx->prompt_text.assign(model_name).append("> ");
    
    std::string msg_mount;
    msg_mount.assign("[Loader] \033[1;32m").append(model_name).append("\033[0m securely bound to Engine.\n");
    shell->ShellPrint(msg_mount);
}

//引入Prompt Prefill和可控采样机制
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
