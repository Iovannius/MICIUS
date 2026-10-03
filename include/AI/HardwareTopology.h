#pragma once

#include <vector>

struct ThreadTopologyPlan {
    int hq_core_id = -1;                //HQ/根管线所在核心
    int recommended_shadow_core = -1;   //为GPU或其他硬件预留的影子核心
    std::vector<int> worker_cores; //Worker线程阵列所在的核心
};

class HardwareTopology
{

public:

    //获取物理核列表
    static std::vector<int> ProbePhysicalCores();
    
    //设置当前线程亲和性
    static void SetThreadAffinity(int cpu_id);

    //将原AIEngineFront中的绑核逻辑抽象为不申请线程的策略规划函数接口
    static ThreadTopologyPlan GenerateDeploymentPlan();
};
