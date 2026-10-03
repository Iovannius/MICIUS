#include "LoggerCore.h"
#include "HardwareTopology.h"

#include <thread>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

namespace micius {
namespace log {

struct AILoggerCore::DaemonState {
    std::thread daemon_thread;
    std::atomic<bool> is_running{false};
    int shadow_core_id = -1;
    FILE* log_file = nullptr;
};

uint64_t AILoggerCore::GetFastNanoTime()
{
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

void AILoggerCore::Start(int shadow_core_id, const char* log_file_path)
{
    if (!log_file_path || log_file_path[0] == '\0') return;
    if (!daemon_state_) {
        daemon_state_ = new DaemonState();
    }
    
    if (daemon_state_->is_running.exchange(true)) return;

    daemon_state_->shadow_core_id = shadow_core_id;
    daemon_state_->log_file = std::fopen(log_file_path, "a");
    if (daemon_state_->log_file) {
        //2MB全缓冲
        static char io_buffer[2 * 1024 * 1024];
        std::setvbuf(daemon_state_->log_file, io_buffer, _IOFBF, sizeof(io_buffer));
    }

    try {
        daemon_state_->daemon_thread = std::thread(&AILoggerCore::BackendDaemonLoop, this);
    } catch (...) {
        daemon_state_->is_running.store(false, std::memory_order_release);
        if (daemon_state_->log_file) {
            std::fclose(daemon_state_->log_file);
            daemon_state_->log_file = nullptr;
        }
        throw;
    }
}

void AILoggerCore::Stop()
{
    if (!daemon_state_) return;

    daemon_state_->is_running.store(false, std::memory_order_release);
    if (daemon_state_->daemon_thread.joinable()) {
        daemon_state_->daemon_thread.join();
    }
    if (daemon_state_->log_file) {
        std::fflush(daemon_state_->log_file);
        std::fclose(daemon_state_->log_file);
        daemon_state_->log_file = nullptr;
    }
    delete daemon_state_;
    daemon_state_ = nullptr;
}

void AILoggerCore::BackendDaemonLoop()
{
    if (daemon_state_->shadow_core_id >= 0) {
        //弃用一开始的设计中硬编码的POSIX接口，复用MICIUS的拓扑接口防止Scheduler与Logger冲突
        HardwareTopology::SetThreadAffinity(daemon_state_->shadow_core_id);
    }

    LogEvent event;
    uint32_t spin_counter = 0;
    //退避阈值，后续可注入LoggerConfig
    const uint32_t SPIN_LIMIT = 2000; 

    //标准Drain Loop
    while (true) {
        if (queue_.Dequeue(event)) {
            spin_counter = 0;
            FormatAndFlush(event);
        } else {
            //队列为空且收到停止信号时退出
            if (!daemon_state_->is_running.load(std::memory_order_relaxed)) {
                break; 
            }
            /*Hybrid Backoff*/
            if (spin_counter < SPIN_LIMIT) {
                _mm_pause(); 
                spin_counter++;
            } else {
                std::this_thread::yield();
            }
        }
    }
}

void AILoggerCore::FormatAndFlush(const LogEvent& event)
{
    if (!daemon_state_ || !daemon_state_->log_file) return;

    const char* level_str = "UNK";
    switch (event.level) {
        case LogLevel::TRACE: level_str = "TRC"; break;
        case LogLevel::DEBUG: level_str = "DBG"; break;
        case LogLevel::INFO:  level_str = "INF"; break;
        case LogLevel::WARN:  level_str = "WRN"; break;
        case LogLevel::ERR:   level_str = "ERR"; break;
        case LogLevel::FATAL: level_str = "FTL"; break;
    }

    //格式化头部信息
    //[Time][Level][Epoch][Thread][File:Line]
    std::fprintf(daemon_state_->log_file, "[%llu][%s][E:%u][T:%u][%s:%u] ", 
                 (unsigned long long)event.timestamp_ns,
                 level_str, event.epoch, event.thread_id, 
                 event.file_name, event.line_num);

    //扫描fmt_str找到占位符并用TypeTag解析而不是透传给printf
    char formatted_msg[1024];
    char* out = formatted_msg;
    const char* in = event.fmt_str;
    size_t remain = sizeof(formatted_msg);
    int arg_idx = 0;
    
    while (*in && remain > 1) {
        if (*in == '%' && *(in + 1) != '%') {
            //跳过格式化占位符
            in++;
            while (*in && std::strchr("-+ #0123456789.hljztL", *in)) in++;
            if (*in) in++; 
            
            //根据保存的TypeTag格式化
            char arg_str[64];
            const ArgType type = arg_idx < 3
                ? static_cast<ArgType>(event.padding[arg_idx])
                : ArgType::NONE;
            if (type == ArgType::FLOAT64) {
                std::snprintf(arg_str, sizeof(arg_str), "%f", event.payload.f64_args[arg_idx]);
            } else if (type == ArgType::INT64) {
                std::snprintf(arg_str, sizeof(arg_str), "%lld", (long long)event.payload.i64_args[arg_idx]);
            } else if (type == ArgType::UINT64) {
                std::snprintf(arg_str, sizeof(arg_str), "%llu", (unsigned long long)event.payload.u64_args[arg_idx]);
            } else {
                std::snprintf(arg_str, sizeof(arg_str), "%s", "<missing>");
            }
            
            size_t len = std::strlen(arg_str);
            if (len < remain) {
                std::memcpy(out, arg_str, len);
                out += len;
                remain -= len;
            }
            if (arg_idx < 3) arg_idx++;
        } else if (*in == '%' && *(in + 1) == '%') {
            //处理转义的%%
            *out++ = '%'; 
            remain--; 
            in += 2;
        } else {
            *out++ = *in++; 
            remain--;
        }
    }
    *out = '\0';

    std::fprintf(daemon_state_->log_file, "%s\n", formatted_msg);
}

} // namespace log
} // namespace micius
