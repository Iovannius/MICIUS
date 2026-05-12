#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstdlib>



/*---------------64字节对齐容器 (填满单条L1D缓存行)---------------*/

struct alignas(64) Chunk16f
{
    float data[16];
};

struct alignas(64) ChunkMask16
{
    uint32_t mask[16]; //32位掩码，0xFFFFFFFF表示有效，0x00000000表示脏数据
};

/*----------------4KB页对齐簇元数据（用于宏观分支预测）---------------*/

#define CHUNKS_PER_CLUSTER 64

struct ClusterMeta
{
    uint32_t active_chunks; //当前4KB簇中存活Chunk数量
    uint32_t is_dirty; //是否主要为脏数据
};

/*---------------全局竞技场化 (Global Arena)---------------*/

class GlobalArena {
public:
    size_t total_clusters;
    size_t total_chunks;
    size_t total_entities;

    Chunk16f* posX;
    Chunk16f* posY;
    Chunk16f* velX;
    Chunk16f* velY;
    ChunkMask16* masks;
    ClusterMeta* meta;

    GlobalArena(size_t entities)
    {

        total_entities = entities;
        total_chunks = (entities + 15) / 16;
        total_clusters = (total_chunks + CHUNKS_PER_CLUSTER - 1) / CHUNKS_PER_CLUSTER;
        
        //重新对齐总Chunk数
        total_chunks = total_clusters * CHUNKS_PER_CLUSTER;

        posX  = (Chunk16f*)_mm_malloc(total_chunks * sizeof(Chunk16f), 64);
        posY  = (Chunk16f*)_mm_malloc(total_chunks * sizeof(Chunk16f), 64);
        velX  = (Chunk16f*)_mm_malloc(total_chunks * sizeof(Chunk16f), 64);
        velY  = (Chunk16f*)_mm_malloc(total_chunks * sizeof(Chunk16f), 64);
        masks = (ChunkMask16*)_mm_malloc(total_chunks * sizeof(ChunkMask16), 64);
        
        meta  = (ClusterMeta*)malloc(total_clusters * sizeof(ClusterMeta));

        //初始注入
        //模拟随机分布与碎片
        for (size_t c = 0; c < total_clusters; ++c) {
            meta[c].active_chunks = CHUNKS_PER_CLUSTER;
            meta[c].is_dirty = 1; //初始标记为脏，首帧强制整理

            for (size_t i = 0; i < CHUNKS_PER_CLUSTER; ++i) {
                size_t global_idx = c * CHUNKS_PER_CLUSTER + i;
                for (int j = 0; j < 16; ++j) {
                    posX[global_idx].data[j] = (float)(rand() % 1000) / 10.0f;
                    posY[global_idx].data[j] = (float)(rand() % 1000) / 10.0f;
                    velX[global_idx].data[j] = 0.0f;
                    velY[global_idx].data[j] = 0.0f;
                    
                    //模拟初始粒子存活率80%
                    masks[global_idx].mask[j] = (rand() % 100 < 80) ? 0xFFFFFFFF : 0x00000000;

                }
            }
        }
    }

    ~GlobalArena()
    {
        _mm_free(posX);
        _mm_free(posY);
        _mm_free(velX);
        _mm_free(velY);
        _mm_free(masks);
        free(meta);
    }

};

/*---------------全局竞技场化 (Global Arena)---------------*/