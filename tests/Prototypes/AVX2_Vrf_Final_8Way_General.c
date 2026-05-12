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
#include <time.h> // 原有的 POSIX 代码保持不变，用于跨平台

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

    //8路展开需要16个外部FP64全局累加器
    //编译器会自动管理Spill，不影响热路径性能
    __m256d g_sum0 = _mm256_setzero_pd();
    __m256d g_sum1 = _mm256_setzero_pd();
    __m256d g_sum2 = _mm256_setzero_pd();
    __m256d g_sum3 = _mm256_setzero_pd();
    __m256d g_sum4 = _mm256_setzero_pd();
    __m256d g_sum5 = _mm256_setzero_pd();
    __m256d g_sum6 = _mm256_setzero_pd();
    __m256d g_sum7 = _mm256_setzero_pd();
    __m256d g_sum8 = _mm256_setzero_pd();
    __m256d g_sum9 = _mm256_setzero_pd();
    __m256d g_sum10 = _mm256_setzero_pd();
    __m256d g_sum11 = _mm256_setzero_pd();
    __m256d g_sum12 = _mm256_setzero_pd();
    __m256d g_sum13 = _mm256_setzero_pd();
    __m256d g_sum14 = _mm256_setzero_pd();
    __m256d g_sum15 = _mm256_setzero_pd();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        //内层局部累加器增加到8个
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        //没有任何条件分支的热路径运算
        //LSD满载的纯FMA运算指令流


        //步进改为64以匹配8路展开
        for (size_t i = 0; i < ARRAY_SIZE; i += 64) {
            
            //数据加载
            __m256 vx0 = _mm256_load_ps(&x[i]);
            __m256 vy0 = _mm256_load_ps(&y[i]);
            __m256 vx1 = _mm256_load_ps(&x[i + 8]);
            __m256 vy1 = _mm256_load_ps(&y[i + 8]);
            __m256 vx2 = _mm256_load_ps(&x[i + 16]);
            __m256 vy2 = _mm256_load_ps(&y[i + 16]);
            __m256 vx3 = _mm256_load_ps(&x[i + 24]);
            __m256 vy3 = _mm256_load_ps(&y[i + 24]);
            __m256 vx4 = _mm256_load_ps(&x[i + 32]);
            __m256 vy4 = _mm256_load_ps(&y[i + 32]);
            __m256 vx5 = _mm256_load_ps(&x[i + 40]);
            __m256 vy5 = _mm256_load_ps(&y[i + 40]);
            __m256 vx6 = _mm256_load_ps(&x[i + 48]);
            __m256 vy6 = _mm256_load_ps(&y[i + 48]);
            __m256 vx7 = _mm256_load_ps(&x[i + 56]);
            __m256 vy7 = _mm256_load_ps(&y[i + 56]);

            //8路计算
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

            __m256 r4 = _mm256_fmadd_ps(vx4, a, vy4);
            r4 = _mm256_fmadd_ps(r4, b, vx4);
            r4 = _mm256_fmadd_ps(r4, c, vy4);
            r4 = _mm256_fmadd_ps(r4, d, a);
            acc4 = _mm256_add_ps(acc4, r4);

            __m256 r5 = _mm256_fmadd_ps(vx5, a, vy5);
            r5 = _mm256_fmadd_ps(r5, b, vx5);
            r5 = _mm256_fmadd_ps(r5, c, vy5);
            r5 = _mm256_fmadd_ps(r5, d, a);
            acc5 = _mm256_add_ps(acc5, r5);

            __m256 r6 = _mm256_fmadd_ps(vx6, a, vy6);
            r6 = _mm256_fmadd_ps(r6, b, vx6);
            r6 = _mm256_fmadd_ps(r6, c, vy6);
            r6 = _mm256_fmadd_ps(r6, d, a);
            acc6 = _mm256_add_ps(acc6, r6);

            __m256 r7 = _mm256_fmadd_ps(vx7, a, vy7);
            r7 = _mm256_fmadd_ps(r7, b, vx7);
            r7 = _mm256_fmadd_ps(r7, c, vy7);
            r7 = _mm256_fmadd_ps(r7, d, a);
            acc7 = _mm256_add_ps(acc7, r7);
        }

        //无分支的冷路径升维：每128次内层循环执行1次
        //使用单指令将Float转为Double喂给全局FP64累加池，不存在吞吐惩罚

        //对应8个acc提取到16个FP64变量
        g_sum0  = _mm256_add_pd(g_sum0,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 0)));
        g_sum1  = _mm256_add_pd(g_sum1,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc0, 1)));
        
        g_sum2  = _mm256_add_pd(g_sum2,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc1, 0)));
        g_sum3  = _mm256_add_pd(g_sum3,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc1, 1)));
        
        g_sum4  = _mm256_add_pd(g_sum4,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc2, 0)));
        g_sum5  = _mm256_add_pd(g_sum5,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc2, 1)));
        
        g_sum6  = _mm256_add_pd(g_sum6,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc3, 0)));
        g_sum7  = _mm256_add_pd(g_sum7,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc3, 1)));

        g_sum8  = _mm256_add_pd(g_sum8,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc4, 0)));
        g_sum9  = _mm256_add_pd(g_sum9,  _mm256_cvtps_pd(_mm256_extractf128_ps(acc4, 1)));

        g_sum10 = _mm256_add_pd(g_sum10, _mm256_cvtps_pd(_mm256_extractf128_ps(acc5, 0)));
        g_sum11 = _mm256_add_pd(g_sum11, _mm256_cvtps_pd(_mm256_extractf128_ps(acc5, 1)));

        g_sum12 = _mm256_add_pd(g_sum12, _mm256_cvtps_pd(_mm256_extractf128_ps(acc6, 0)));
        g_sum13 = _mm256_add_pd(g_sum13, _mm256_cvtps_pd(_mm256_extractf128_ps(acc6, 1)));

        g_sum14 = _mm256_add_pd(g_sum14, _mm256_cvtps_pd(_mm256_extractf128_ps(acc7, 0)));
        g_sum15 = _mm256_add_pd(g_sum15, _mm256_cvtps_pd(_mm256_extractf128_ps(acc7, 1)));
    }

    double end = get_time();
    double delta = end - start;

    //吞吐量计算
    double total_ops = (double)ARRAY_SIZE * ITERATIONS * 8;
    double gflops = (total_ops / delta) / 1e9;

    //FP64规约归并

    //汇集16个变量
    double simd_sum = 0.0;
    double tmp[4];
    _mm256_storeu_pd(tmp, g_sum0);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum1);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum2);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum3);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum4);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum5);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum6);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum7);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum8);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum9);  simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum10); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum11); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum12); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum13); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum14); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, g_sum15); simd_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];

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