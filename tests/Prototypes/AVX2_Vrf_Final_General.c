#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>

#define ARRAY_SIZE (1024 * 4)
#define ITERATIONS 2000000

#ifdef _WIN32
#include <windows.h>

double get_time()
{
    LARGE_INTEGER count, freq;
    QueryPerformanceCounter(&count);
    QueryPerformanceFrequency(&freq);
    return (double)count.QuadPart / freq.QuadPart;
}

#else
// 原有的 POSIX 代码保持不变，用于跨平台
#include <time.h>
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

int main()
{

    float *x = (float *)_mm_malloc(ARRAY_SIZE * sizeof(float), 64);
    float *y = (float *)_mm_malloc(ARRAY_SIZE * sizeof(float), 64);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = (float)i * 0.0001f;
        y[i] = (float)i * 0.0002f;
    }

    __m256 a = _mm256_set1_ps(1.1f);
    __m256 b = _mm256_set1_ps(1.2f);
    __m256 c = _mm256_set1_ps(1.3f);
    __m256 d = _mm256_set1_ps(1.4f);

    /*HPC部分*/
    double start = get_time();

    //求和使用的外部FP64全局累加器
    //此处实现是常驻化利用8个YMM寄存器，尽可能消除写回内存的性能干扰
    __m256d g_sum0 = _mm256_setzero_pd();
    __m256d g_sum1 = _mm256_setzero_pd();
    __m256d g_sum2 = _mm256_setzero_pd();
    __m256d g_sum3 = _mm256_setzero_pd();
    __m256d g_sum4 = _mm256_setzero_pd();
    __m256d g_sum5 = _mm256_setzero_pd();
    __m256d g_sum6 = _mm256_setzero_pd();
    __m256d g_sum7 = _mm256_setzero_pd();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        //分层设计的FP32累加树是本架构的重点之一
        //内层局部累加器，每次iter清零
        //它只负责累加128次，尽量保证FP32不出现精度问题
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        //没有任何条件分支的热路径运算
        //LSD满载的纯FMA运算指令流
        for (size_t i = 0; i < ARRAY_SIZE; i += 32) {
            __m256 vx0 = _mm256_load_ps(&x[i]);
            __m256 vy0 = _mm256_load_ps(&y[i]);
            __m256 vx1 = _mm256_load_ps(&x[i + 8]);
            __m256 vy1 = _mm256_load_ps(&y[i + 8]);
            __m256 vx2 = _mm256_load_ps(&x[i + 16]);
            __m256 vy2 = _mm256_load_ps(&y[i + 16]);
            __m256 vx3 = _mm256_load_ps(&x[i + 24]);
            __m256 vy3 = _mm256_load_ps(&y[i + 24]);

            __m256 r0 = _mm256_fmadd_ps(vx0, a, vy0);
            r0 = _mm256_fmadd_ps(r0, b, vx0);
            r0 = _mm256_fmadd_ps(r0, c, vy0);
            r0 = _mm256_fmadd_ps(r0, d, a);
            acc0 = _mm256_add_ps(acc0, r0);

            __m256 r1 = _mm256_fmadd_ps(vx1, a, vy1);
            r1 = _mm256_fmadd_ps(r1, b, vx1);
            r1 = _mm256_fmadd_ps(r1, c, vy1);
            r1 = _mm256_fmadd_ps(r1, d, a);
            acc1 = _mm256_add_ps(acc1, r1);

            __m256 r2 = _mm256_fmadd_ps(vx2, a, vy2);
            r2 = _mm256_fmadd_ps(r2, b, vx2);
            r2 = _mm256_fmadd_ps(r2, c, vy2);
            r2 = _mm256_fmadd_ps(r2, d, a);
            acc2 = _mm256_add_ps(acc2, r2);

            __m256 r3 = _mm256_fmadd_ps(vx3, a, vy3);
            r3 = _mm256_fmadd_ps(r3, b, vx3);
            r3 = _mm256_fmadd_ps(r3, c, vy3);
            r3 = _mm256_fmadd_ps(r3, d, a);
            acc3 = _mm256_add_ps(acc3, r3);
        }

        //无分支的冷路径升维：每128次内层循环执行1次
        //使用单指令将Float转为Double喂给全局FP64累加池，不存在吞吐惩罚
        g_sum0 = _mm256_add_pd(g_sum0, _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 0)));
        g_sum1 = _mm256_add_pd(g_sum1, _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 1)));
        
        g_sum2 = _mm256_add_pd(g_sum2, _mm256_cvtps_pd(_mm256_extractf128_ps(acc1, 0)));
        g_sum3 = _mm256_add_pd(g_sum3, _mm256_cvtps_pd(_mm256_extractf128_ps(acc1, 1)));
        
        g_sum4 = _mm256_add_pd(g_sum4, _mm256_cvtps_pd(_mm256_extractf128_ps(acc2, 0)));
        g_sum5 = _mm256_add_pd(g_sum5, _mm256_cvtps_pd(_mm256_extractf128_ps(acc2, 1)));
        
        g_sum6 = _mm256_add_pd(g_sum6, _mm256_cvtps_pd(_mm256_extractf128_ps(acc3, 0)));
        g_sum7 = _mm256_add_pd(g_sum7, _mm256_cvtps_pd(_mm256_extractf128_ps(acc3, 1)));
    }

    double end = get_time();
    double delta = end - start;

    //吞吐量计算
    double total_ops = (double)ARRAY_SIZE * ITERATIONS * 8;
    double gflops = (total_ops / delta) / 1e9;

    //FP64规约归并
    double simd_sum = 0.0;
    double tmp[4];
    _mm256_storeu_pd(tmp, g_sum0); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum1); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum2); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum3); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum4); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum5); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum6); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum7); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];

    printf("---------------------------------------\n");
    printf("Time        : %.4f s\n", delta);
    printf("Throughput  : %.2f GFLOPS\n", gflops);
    printf("SIMD sum    : %.10e\n", simd_sum);
    printf("---------------------------------------\n");

    //单次验证
    //强迫Reference也在FP32环境下完成4096次累加以匹配SIMD的物理动作，消除跨精度对比产生的硬件漂移
    float ref_float_sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        float r = x[i] * 1.1f + y[i];
        r = r * 1.2f + x[i];
        r = r * 1.3f + y[i];
        r = r * 1.4f + 1.1f;
        ref_float_sum += r;
    }
    double target_ref_sum = (double)ref_float_sum * ITERATIONS;

    //SIMD自身的单次检查也要对齐FP32累加树
    float simd_check_float = 0.0f;
    for (size_t i = 0; i < ARRAY_SIZE; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 vy = _mm256_load_ps(&y[i]);

        __m256 r = _mm256_fmadd_ps(vx, a, vy);
        r = _mm256_fmadd_ps(r, b, vx);
        r = _mm256_fmadd_ps(r, c, vy);
        r = _mm256_fmadd_ps(r, d, a);

        float t[8];
        _mm256_store_ps(t, r);
        for (int j = 0; j < 8; j++) simd_check_float += t[j];
    }
    double simd_check = (double)simd_check_float * ITERATIONS;

    double diff = fabs(target_ref_sum - simd_sum);
    double rel_error = diff / fabs(target_ref_sum);

    printf("Ref sum     : %.10e\n", target_ref_sum);
    printf("SIMD check  : %.10e\n", simd_check);
    printf("Diff        : %.10e\n", diff);
    printf("RelErr      : %.10e\n", rel_error);

    if (rel_error < 1e-6)
        printf("PASS\n");
    else
        printf("FAIL\n");

    _mm_free(x);
    _mm_free(y);
    return 0;
}