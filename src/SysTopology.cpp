

#include <iostream>
#include <vector>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#include <pthread.h>
#include <fstream>
#include <string>
#include <set>
#endif



#ifdef _WIN32
double get_time()
{
    LARGE_INTEGER count, freq;
    QueryPerformanceCounter(&count);
    QueryPerformanceFrequency(&freq);
    return (double)count.QuadPart / freq.QuadPart;
}
#else
#include <time.h>
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif



std::vector<int> get_physical_core_ids()
{

    std::vector<int> physical_cores;

#ifdef _WIN32
    //Windows端使用GetLogicalProcessorInformation探测真实的芯片拓扑
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buffer.data(), &length)) {
            for (const auto& info : buffer) {
                //只关注处理器核心级别的拓扑关系
                if (info.Relationship == RelationProcessorCore) {
                    //ProcessorMask包含了这个物理核心上的所有逻辑线程
                    //比如超线程开启时，可能是0011，代表线程 0 和 1
                    ULONG_PTR mask = info.ProcessorMask;
                    
                    //核心算法：提取最低位的1，剥离掉第二个超线程
                    ULONG_PTR lowest_bit_mask = mask & ((~mask) + 1);
                    
                    //将Mask转换成具体的CPUID
                    int cpu_index = 0;
                    while ((lowest_bit_mask & 1) == 0) {
                        lowest_bit_mask >>= 1;
                        cpu_index++;
                    }
                    physical_cores.push_back(cpu_index);
                }
            }
        }
    }
#else
    //Linux端解析sysfs拓扑树，避免简单的奇偶猜测
    int num_logical = std::thread::hardware_concurrency();
    std::set<std::string> seen_cores;

    for (int i = 0; i < num_logical; ++i) {
        std::string phys_pkg_path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/physical_package_id";
        std::string core_id_path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id";

        std::ifstream pkg_file(phys_pkg_path);
        std::ifstream core_file(core_id_path);

        if (pkg_file.is_open() && core_file.is_open()) {
            std::string pkg, core;
            pkg_file >> pkg;
            core_file >> core;
            std::string unique_id = pkg + "_" + core;

            //如果这个核心第一次见到，说明当前逻辑CPU是它的第一个线程
            if (seen_cores.find(unique_id) == seen_cores.end()) {
                seen_cores.insert(unique_id);
                physical_cores.push_back(i);
            }
        }
    }
#endif

    return physical_cores;
}

//绑核函数封装
void pin_thread_to_cpu(int cpu_id)
{

#ifdef _WIN32
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << cpu_id);
    SetThreadAffinityMask(thread, mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

}