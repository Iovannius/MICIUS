#include "HardwareTopology.h"

#include <algorithm>
#include <iostream>
#include <limits>
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

void HardwareTopology::SetThreadAffinity(int cpu_id)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

std::vector<int> HardwareTopology::ProbePhysicalCores()
{
    std::vector<int> physical_cores;
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buffer.data(), &length)) {
            for (const auto& info : buffer) {
                if (info.Relationship == RelationProcessorCore) {
                    ULONG_PTR mask = info.ProcessorMask;
                    ULONG_PTR lowest_bit = mask & ((~mask) + 1);
                    int idx = 0;
                    while ((lowest_bit & 1) == 0) {
                        lowest_bit >>= 1;
                        idx++;
                    }
                    physical_cores.push_back(idx);
                }
            }
        }
    }
#else
    const unsigned int detected_logical = std::thread::hardware_concurrency();
    const int num_logical = detected_logical > static_cast<unsigned int>(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max()
        : static_cast<int>(detected_logical);
    std::set<std::string> seen_cores;
    for (int i = 0; i < num_logical; ++i) {
        std::ifstream pkg("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/physical_package_id");
        std::ifstream core("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id");
        if (pkg.is_open() && core.is_open()) {
            std::string p, c; pkg >> p; core >> c;
            if (seen_cores.insert(p + "_" + c).second) physical_cores.push_back(i);
        }
    }
#endif
    return physical_cores;
}

ThreadTopologyPlan HardwareTopology::GenerateDeploymentPlan()
{
    ThreadTopologyPlan plan;
    std::vector<int> phys_cores = ProbePhysicalCores();
    const unsigned int detected_logical = std::thread::hardware_concurrency();
    const int total_logical_cores =
        detected_logical > static_cast<unsigned int>(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max()
        : static_cast<int>(detected_logical);
    const size_t logical_count = static_cast<size_t>(std::max(total_logical_cores, 0));
    const bool is_hetero =
        logical_count != phys_cores.size() && logical_count != phys_cores.size() * 2U;

    if (phys_cores.size() < 2) {
        int fallback_logical_count = total_logical_cores;
        if (fallback_logical_count < 2) fallback_logical_count = 2;
        phys_cores.clear();
        for (int i = 0; i < fallback_logical_count; ++i) phys_cores.push_back(i);
        std::cout << "[Kernel] OS Topology Probe failed. Fallback to " << phys_cores.size() << " logical cores.\n";
        
        plan.hq_core_id = phys_cores[0];
        plan.recommended_shadow_core = (phys_cores.size() > 1) ? phys_cores[1] : phys_cores[0];
        for (size_t i = 1; i < phys_cores.size(); ++i) {
            if (phys_cores[i] != plan.recommended_shadow_core) plan.worker_cores.push_back(phys_cores[i]);
        }
    } else if (phys_cores.size() == 2) {
        plan.hq_core_id = phys_cores[0]; 
        plan.recommended_shadow_core = -1;
        for (int id = 0; id < total_logical_cores; ++id) {
            bool is_phys_head = false;
            for (int p_core : phys_cores) { if (p_core == id) is_phys_head = true; }
            if (!is_phys_head) {
                plan.recommended_shadow_core = id;
                break;
            }
        }
        if (plan.recommended_shadow_core == -1) plan.recommended_shadow_core = plan.hq_core_id + 1;
        plan.worker_cores.push_back(phys_cores[1]);
    } else if (is_hetero) {
        const int physical_count = static_cast<int>(phys_cores.size());
        int num_p_cores = total_logical_cores - physical_count;
        int total_e_cores = physical_count - num_p_cores;

        plan.hq_core_id = phys_cores[0];              
        plan.recommended_shadow_core = phys_cores[1]; 
        int valid_e_cores = total_e_cores;
        if (total_e_cores % 4 == 2) {
            valid_e_cores -= 2; 
            std::cout << "[Kernel] Meteor Lake detected. Dropping 2 SoC LP-E cores.\n";
        }
        for (int i = 0; i < valid_e_cores; ++i) {
            plan.worker_cores.push_back(phys_cores[num_p_cores + i]); 
        }
    } else {
        plan.hq_core_id = phys_cores[0];                 
        plan.recommended_shadow_core = phys_cores.back();
        for (size_t i = 1; i < phys_cores.size() - 1; ++i) {
            plan.worker_cores.push_back(phys_cores[i]);  
        }
    }
    return plan;
}
