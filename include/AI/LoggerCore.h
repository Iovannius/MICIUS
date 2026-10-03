#pragma once

#include <cstdint>
#include <atomic>
#include <chrono>
#include <type_traits>

namespace micius {
namespace log {

//消息类型枚举
enum class LogLevel : uint8_t { TRACE = 0, DEBUG, INFO, WARN, ERR, FATAL };
enum class ArgType : uint8_t { NONE = 0, INT64, UINT64, FLOAT64 };

struct alignas(64) LogEvent {
    uint64_t timestamp_ns;
    const char* file_name;
    const char* fmt_str;
    uint32_t line_num;
    uint32_t thread_id;
    uint32_t epoch;
    LogLevel level;
    uint8_t  padding[3]; 
    
    union {
        uint64_t u64_args[3];
        double   f64_args[3];
        int64_t  i64_args[3];
    } payload;
};
static_assert(sizeof(LogEvent) == 64, "LogEvent must be exactly 64 bytes.");

template<typename T>
constexpr ArgType GetArgType()
{
    if constexpr (std::is_floating_point_v<T>) return ArgType::FLOAT64;
    else if constexpr (std::is_signed_v<T>)    return ArgType::INT64;
    else if constexpr (std::is_unsigned_v<T> || std::is_pointer_v<T>) return ArgType::UINT64;
    else return ArgType::NONE;
}

template<size_t Capacity>
class AsyncLogQueue
{

    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    struct alignas(64) Cell
    {
        std::atomic<size_t> sequence;
        LogEvent data;
    };

    alignas(64) Cell buffer_[Capacity];
    alignas(64) std::atomic<size_t> enqueue_pos_;
    alignas(64) std::atomic<size_t> dequeue_pos_;
    size_t buffer_mask_;

public:

    AsyncLogQueue()
        : buffer_mask_(Capacity - 1)
    {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    bool Enqueue(const LogEvent& event)
    {
        Cell* cell = nullptr;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & buffer_mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break; 
            } else if (diff < 0) {
                return false; 
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        cell->data = event;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool Dequeue(LogEvent& out_event) {
        Cell* cell = nullptr;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & buffer_mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        out_event = cell->data;
        cell->sequence.store(pos + buffer_mask_ + 1, std::memory_order_release);
        return true;
    }
};

//日志系统前端
class AILoggerCore
{
public:

    static AILoggerCore& GetInstance()
    {
        static AILoggerCore instance;
        return instance;
    }

    void Start(int shadow_core_id, const char* log_file_path);
    void Stop();
    static uint64_t GetFastNanoTime();

    template<typename... Args>
    void CommitLog(LogLevel level, const char* file, uint32_t line, 
                   uint32_t epoch, uint32_t thread_id, const char* fmt, Args... args)
    {
        
        static_assert(sizeof...(Args) <= 3, "MICIUS Logger supports max 3 arguments.");

        LogEvent event{};
        event.timestamp_ns = GetFastNanoTime();
        event.file_name = file;
        event.fmt_str = fmt;
        event.line_num = line;
        event.thread_id = thread_id;
        event.epoch = epoch;
        event.level = level;
        event.padding[0] = static_cast<uint8_t>(ArgType::NONE);
        event.padding[1] = static_cast<uint8_t>(ArgType::NONE);
        event.padding[2] = static_cast<uint8_t>(ArgType::NONE);

        int index = 0;
        ([&] {
            using T = std::decay_t<Args>;
            constexpr ArgType type = GetArgType<T>();
            event.padding[index] = static_cast<uint8_t>(type);
            
            if constexpr (type == ArgType::FLOAT64) {
                event.payload.f64_args[index] = static_cast<double>(args);
            } else if constexpr (type == ArgType::INT64) {
                event.payload.i64_args[index] = static_cast<int64_t>(args);
            } else if constexpr (std::is_pointer_v<T>) {
                event.payload.u64_args[index] =
                    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args));
            } else if constexpr (type == ArgType::UINT64) {
                event.payload.u64_args[index] = static_cast<uint64_t>(args);
            } else {
                event.payload.u64_args[index] = 0;
            }
            index++;
        }(), ...);

        (void)queue_.Enqueue(event);
    }

private:
    AILoggerCore() = default;
    ~AILoggerCore() { Stop(); }

    void BackendDaemonLoop();
    void FormatAndFlush(const LogEvent& event);

    struct DaemonState; 
    DaemonState* daemon_state_ = nullptr;
    
    AsyncLogQueue<65536> queue_; 
};

//暴露给Worker的宏
#define MICIUS_KRNL_LOG(level, epoch, fmt, ...) \
    ::micius::log::AILoggerCore::GetInstance().CommitLog( \
        level, __FILE__, __LINE__, epoch, 0, fmt, ##__VA_ARGS__)

#define MICIUS_KRNL_INFO(epoch, fmt, ...)  MICIUS_KRNL_LOG(::micius::log::LogLevel::INFO, epoch, fmt, ##__VA_ARGS__)
#define MICIUS_KRNL_ERROR(epoch, fmt, ...) MICIUS_KRNL_LOG(::micius::log::LogLevel::ERR, epoch, fmt, ##__VA_ARGS__)

} // namespace log
} // namespace micius
