#include <iostream>
#include <vector>
#include <thread>
#include <immintrin.h>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#include <pthread.h>
#include <fstream>
#include <string>
#include <set>
#endif

#define ARRAY_SIZE (1024 * 3) //12KB, 塞入单核L1D
#define ITERATIONS 2000000

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
//------------------------------物理核心数量探测及绑定------------------------------
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
//------------------------------物理核心数量探测及绑定------------------------------
void worker_thread(int cpu_id, double* out_gflops, double* out_simd_sum, double* out_ref_sum, double* out_simd_check)
{
    
    pin_thread_to_cpu(cpu_id);

    //线程内部独立分配内存，确保物理内存映射到执行该线程的核心上
    //NUMA及L1友好😼
    float *x = (float *)_mm_malloc(ARRAY_SIZE * sizeof(float), 64);
    float *y = (float *)_mm_malloc(ARRAY_SIZE * sizeof(float), 64);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = (float)i * 0.0001f + cpu_id; //加一点cpu_id产生微小差异
        y[i] = (float)i * 0.0002f + cpu_id;
    }

    __m256 a = _mm256_set1_ps(1.1f);
    __m256 b = _mm256_set1_ps(1.2f);
    __m256 c = _mm256_set1_ps(1.3f);
    __m256 d = _mm256_set1_ps(1.4f);

    __m256d g_sum[16];
    for(int i=0; i<16; ++i) g_sum[i] = _mm256_setzero_pd();

    double start = get_time();

    //8路展开的核心热路径
    for (int iter = 0; iter < ITERATIONS; iter++) {

        __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps(); __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps(); __m256 acc7 = _mm256_setzero_ps();

        for (size_t i = 0; i < ARRAY_SIZE; i += 64) {

            __m256 vx0 = _mm256_load_ps(&x[i]);
            __m256 vy0 = _mm256_load_ps(&y[i]);

            __m256 vx1 = _mm256_load_ps(&x[i+8]);
            __m256 vy1 = _mm256_load_ps(&y[i+8]);

            __m256 vx2 = _mm256_load_ps(&x[i+16]);
            __m256 vy2 = _mm256_load_ps(&y[i+16]);

            __m256 vx3 = _mm256_load_ps(&x[i+24]);
            __m256 vy3 = _mm256_load_ps(&y[i+24]);

            __m256 vx4 = _mm256_load_ps(&x[i+32]);
            __m256 vy4 = _mm256_load_ps(&y[i+32]);

            __m256 vx5 = _mm256_load_ps(&x[i+40]);
            __m256 vy5 = _mm256_load_ps(&y[i+40]);

            __m256 vx6 = _mm256_load_ps(&x[i+48]);
            __m256 vy6 = _mm256_load_ps(&y[i+48]);

            __m256 vx7 = _mm256_load_ps(&x[i+56]);
            __m256 vy7 = _mm256_load_ps(&y[i+56]);

            acc0 = _mm256_fmadd_ps(vx0, a, acc0);
            acc0 = _mm256_fmadd_ps(vy0, b, acc0);
            acc0 = _mm256_fmadd_ps(vx0, c, acc0);
            acc0 = _mm256_fmadd_ps(vy0, d, acc0);

            acc1 = _mm256_fmadd_ps(vx1, a, acc1);
            acc1 = _mm256_fmadd_ps(vy1, b, acc1);
            acc1 = _mm256_fmadd_ps(vx1, c, acc1);
            acc1 = _mm256_fmadd_ps(vy1, d, acc1);

            acc2 = _mm256_fmadd_ps(vx2, a, acc2);
            acc2 = _mm256_fmadd_ps(vy2, b, acc2);
            acc2 = _mm256_fmadd_ps(vx2, c, acc2);
            acc2 = _mm256_fmadd_ps(vy2, d, acc2);
            
            acc3 = _mm256_fmadd_ps(vx3, a, acc3);
            acc3 = _mm256_fmadd_ps(vy3, b, acc3);
            acc3 = _mm256_fmadd_ps(vx3, c, acc3);
            acc3 = _mm256_fmadd_ps(vy3, d, acc3);

            acc4 = _mm256_fmadd_ps(vx4, a, acc4);
            acc4 = _mm256_fmadd_ps(vy4, b, acc4);
            acc4 = _mm256_fmadd_ps(vx4, c, acc4);
            acc4 = _mm256_fmadd_ps(vy4, d, acc4);

            acc5 = _mm256_fmadd_ps(vx5, a, acc5);
            acc5 = _mm256_fmadd_ps(vy5, b, acc5);
            acc5 = _mm256_fmadd_ps(vx5, c, acc5);
            acc5 = _mm256_fmadd_ps(vy5, d, acc5);

            acc6 = _mm256_fmadd_ps(vx6, a, acc6);
            acc6 = _mm256_fmadd_ps(vy6, b, acc6);
            acc6 = _mm256_fmadd_ps(vx6, c, acc6);
            acc6 = _mm256_fmadd_ps(vy6, d, acc6);

            acc7 = _mm256_fmadd_ps(vx7, a, acc7);
            acc7 = _mm256_fmadd_ps(vy7, b, acc7);
            acc7 = _mm256_fmadd_ps(vx7, c, acc7);
            acc7 = _mm256_fmadd_ps(vy7, d, acc7);

        }

        //FP64升维归并
        g_sum[0]  = _mm256_add_pd(g_sum[0],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 0)));
        g_sum[1]  = _mm256_add_pd(g_sum[1],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 1)));
        g_sum[2]  = _mm256_add_pd(g_sum[2],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc1, 0)));
        g_sum[3]  = _mm256_add_pd(g_sum[3],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc1, 1)));
        g_sum[4]  = _mm256_add_pd(g_sum[4],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc2, 0)));
        g_sum[5]  = _mm256_add_pd(g_sum[5],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc2, 1)));
        g_sum[6]  = _mm256_add_pd(g_sum[6],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc3, 0)));
        g_sum[7]  = _mm256_add_pd(g_sum[7],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc3, 1)));
        g_sum[8]  = _mm256_add_pd(g_sum[8],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc4, 0)));
        g_sum[9]  = _mm256_add_pd(g_sum[9],  _mm256_cvtps_pd(_mm256_extractf128_ps(acc4, 1)));
        g_sum[10] = _mm256_add_pd(g_sum[10], _mm256_cvtps_pd(_mm256_extractf128_ps(acc5, 0)));
        g_sum[11] = _mm256_add_pd(g_sum[11], _mm256_cvtps_pd(_mm256_extractf128_ps(acc5, 1)));
        g_sum[12] = _mm256_add_pd(g_sum[12], _mm256_cvtps_pd(_mm256_extractf128_ps(acc6, 0)));
        g_sum[13] = _mm256_add_pd(g_sum[13], _mm256_cvtps_pd(_mm256_extractf128_ps(acc6, 1)));
        g_sum[14] = _mm256_add_pd(g_sum[14], _mm256_cvtps_pd(_mm256_extractf128_ps(acc7, 0)));
        g_sum[15] = _mm256_add_pd(g_sum[15], _mm256_cvtps_pd(_mm256_extractf128_ps(acc7, 1)));

    }

    double end = get_time();
    double delta = end - start;

    double total_ops = (double)ARRAY_SIZE * ITERATIONS * 8;
    *out_gflops = (total_ops / delta) / 1e9;

    double thread_simd_sum = 0.0;
    double tmp[4];
    for(int i=0; i<16; ++i) {
        _mm256_storeu_pd(tmp, g_sum[i]);
        thread_simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    *out_simd_sum = thread_simd_sum;

    //校验逻辑开始
    //标准参考计算，匹配打断依赖链后的数学等价式
    float ref_float_sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        float r = 0.0f;
        r += x[i] * 1.1f;
        r += y[i] * 1.2f;
        r += x[i] * 1.3f;
        r += y[i] * 1.4f;
        ref_float_sum += r;
    }
    *out_ref_sum = (double)ref_float_sum * ITERATIONS;

    //单次SIMD物理动作校验
    float simd_check_float = 0.0f;
    for (size_t i = 0; i < ARRAY_SIZE; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 vy = _mm256_load_ps(&y[i]);

        __m256 r = _mm256_setzero_ps();
        r = _mm256_fmadd_ps(vx, a, r);
        r = _mm256_fmadd_ps(vy, b, r);
        r = _mm256_fmadd_ps(vx, c, r);
        r = _mm256_fmadd_ps(vy, d, r);

        float t[8];
        _mm256_storeu_ps(t, r);
        for (int j = 0; j < 8; j++) simd_check_float += t[j];
    }
    *out_simd_check = (double)simd_check_float * ITERATIONS;

    _mm_free(x);
    _mm_free(y);
}

int main()
{
    std::vector<int> phys_cores = get_physical_core_ids();
    int num_threads = phys_cores.size();

    std::cout << "Cores detected: " << num_threads << std::endl;
    std::cout << '\n';

    std::vector<std::thread> threads;
    std::vector<double> gflops_results(num_threads, 0.0);
    std::vector<double> sum_results(num_threads, 0.0);

    std::vector<double> ref_results(num_threads, 0.0);
    std::vector<double> check_results(num_threads, 0.0);

    double global_start = get_time();

    //只针对物理核心启动线程
    for (int i = 0; i < num_threads; ++i) {
        int target_cpu = phys_cores[i];
        threads.emplace_back(worker_thread, target_cpu, &gflops_results[i], &sum_results[i], &ref_results[i], &check_results[i]);
    }

    double total_gflops = 0.0;
    double final_sum = 0.0;
    double final_ref_sum = 0.0;
    double final_check_sum = 0.0;

    //线程阻塞与全局汇总
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
        total_gflops += gflops_results[i];
        final_sum += sum_results[i];
        final_ref_sum += ref_results[i];
        final_check_sum += check_results[i];
        printf("Thread %d Throughput: %.2f GFLOPS\n", i, gflops_results[i]);
    }

    std::cout << '\n';

    double global_end = get_time();

    double diff = std::abs(final_ref_sum - final_sum);
    double rel_error = diff / std::abs(final_ref_sum);
    
    printf("---------------------------------------\n");
    printf("Time            : %.4f s\n", global_end - global_start);
    printf("TOTAL Throughput: %.2f GFLOPS\n", total_gflops);
    printf("---------------------------------------\n");
    printf("Ref sum         : %.10e\n", final_ref_sum);
    printf("SIMD check      : %.10e\n", final_check_sum);
    printf("AGGR sum        : %.10e\n", final_sum);
    printf("Diff            : %.10e\n", diff);
    printf("RelErr          : %.10e\n", rel_error);

    if (rel_error < 1e-6) {
        printf("\n>>>> STATUS: PASS <<<<\n\n");
    } else {
        printf("\n>>>> STATUS: FAIL <<<<\n\n");
    }

    return 0;
}