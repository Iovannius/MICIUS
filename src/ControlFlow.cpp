#include "MemoryPool.h"
#include <immintrin.h>
#include <cstring>

//L1D热复用整流
void local_micro_compact(GlobalArena* arena, size_t cluster_idx) {
    size_t base = cluster_idx * CHUNKS_PER_CLUSTER;
    size_t left = base;
    size_t right = base + arena->meta[cluster_idx].active_chunks - 1;

    //判定：256位寄存器直接吞吐测试
    auto is_chunk_dead = [&](size_t idx) -> bool {
        __m256i m0 = _mm256_load_si256((__m256i*)&arena->masks[idx].mask[0]);
        __m256i m1 = _mm256_load_si256((__m256i*)&arena->masks[idx].mask[8]);
        __m256i combined = _mm256_or_si256(m0, m1);
        return _mm256_testz_si256(combined, combined) == 1; 
    };

    int active_count = arena->meta[cluster_idx].active_chunks;
    bool did_compact = false;

    while (left < right) {
        while (left < right && !is_chunk_dead(left)) left++;
        while (left < right && is_chunk_dead(right)) { right--; active_count--; did_compact = true; }

        if (left < right) {
            //64字节对齐下的编译器激进优化
            //（几乎全是VMOVAPS）
            arena->posX[left] = arena->posX[right];
            arena->posY[left] = arena->posY[right];
            arena->velX[left] = arena->velX[right];
            arena->velY[left] = arena->velY[right];
            arena->masks[left] = arena->masks[right];
            
            //抹除原死区痕迹
            _mm256_store_si256((__m256i*)&arena->masks[right].mask[0], _mm256_setzero_si256());
            _mm256_store_si256((__m256i*)&arena->masks[right].mask[8], _mm256_setzero_si256());

            right--;
            active_count--;
            did_compact = true;

        }
    }

    //更新簇元数据
    arena->meta[cluster_idx].active_chunks = active_count;
    arena->meta[cluster_idx].is_dirty = 0;

}


//整理有效Cluster间的碎片
//建议在引擎主循环中每隔60~120帧一次
//或在异步线程中低频调用。
void global_cluster_gc(GlobalArena* arena)
{

    size_t left_c = 0;
    size_t right_c = arena->total_clusters - 1;

    while (left_c < right_c) {
        //左侧：极度稀疏的簇，低于10%存活率
        while (left_c < right_c && arena->meta[left_c].active_chunks > (CHUNKS_PER_CLUSTER / 10)) {
            left_c++;
        }
        
        //右侧：饱满或有复用价值的簇
        while (left_c < right_c && arena->meta[right_c].active_chunks <= (CHUNKS_PER_CLUSTER / 10)) {
            //废弃无药可救的边缘死簇
            arena->meta[right_c].active_chunks = 0; 
            right_c--;
        }

        if (left_c < right_c) {
            //4KB
            //触发总线级搬运，类似磁盘逻辑
            size_t dest_base = left_c * CHUNKS_PER_CLUSTER;
            size_t src_base  = right_c * CHUNKS_PER_CLUSTER;
            size_t bytes_to_move = CHUNKS_PER_CLUSTER * sizeof(Chunk16f);

            //4KB对齐强拷
            std::memcpy(&arena->posX[dest_base],  &arena->posX[src_base],  bytes_to_move);
            std::memcpy(&arena->posY[dest_base],  &arena->posY[src_base],  bytes_to_move);
            std::memcpy(&arena->velX[dest_base],  &arena->velX[src_base],  bytes_to_move);
            std::memcpy(&arena->velY[dest_base],  &arena->velY[src_base],  bytes_to_move);
            std::memcpy(&arena->masks[dest_base], &arena->masks[src_base], CHUNKS_PER_CLUSTER * sizeof(ChunkMask16));

            //元数据继承与销毁
            arena->meta[left_c].active_chunks = arena->meta[right_c].active_chunks;
            arena->meta[left_c].is_dirty = 1;
            
            //强搬运后，建议左侧下次重扫一次
            
            arena->meta[right_c].active_chunks = 0;
            arena->meta[right_c].is_dirty = 0;
            
            right_c--;

        }
    }
}