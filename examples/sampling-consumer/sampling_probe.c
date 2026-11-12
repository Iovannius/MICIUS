#include "micius_sampling.h"
#include <stdio.h>
int main(void) {
    MiciusSamplingOptionsV1 options;
    float logits[]={1,3,3}; uint64_t rng=0; uint32_t token=99;
    if(!Micius_InitSamplingV1(&options,sizeof(options)))return 1;
    if(!Micius_SampleTokenV1(&options,logits,3,0,0,0,&rng,&token) || token!=1 || rng!=0)return 1;
    options.temperature=1; options.top_k=1;
    if(!Micius_SampleTokenV1(&options,logits,3,0,0,0,&rng,&token) || token!=1 || rng==0)return 1;
    puts("Independent sampler C linkage: PASS");return 0;
}
