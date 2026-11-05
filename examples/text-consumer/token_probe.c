#include "micius_text.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Standalone C tokenizer client. It never creates an engine or binds weights. */
static int check(const MiciusTokenizer* tok, const char* input, size_t size, FILE* output)
{
    size_t count=0;
    int rc=Micius_EncodeTextV1(tok,input,size,NULL,0,&count);
    if ((!size && rc!=1) || (size && (rc!=0 || !count)) || count>size) return 0;
    uint32_t* ids=(uint32_t*)calloc(count ? count : 1,sizeof(*ids));
    if (!ids) return 0;
    size_t required=0;
    if (count) {
        ids[0]=UINT32_MAX;
        rc=Micius_EncodeTextV1(tok,input,size,ids,count-1,&required);
        if (rc!=0 || required!=count || ids[0]!=UINT32_MAX) { free(ids); return 0; }
    }
    rc=Micius_EncodeTextV1(tok,input,size,ids,count,&required);
    size_t pos=0;
    int ok=rc==1 && required==count;
    for (size_t i=0; ok && i<count; ++i) {
        const char* bytes=NULL; size_t n=0;
        ok=Micius_TokenBytesV1(tok,ids[i],&bytes,&n)==1 && n && n<=size-pos &&
           memcmp(input+pos,bytes,n)==0;
        if (ok) pos+=n;
    }
    ok=ok && pos==size && fwrite(ids,sizeof(*ids),count,output)==count;
    free(ids);
    if (ok) printf("tokens=%zu\nroundtrip=PASS\n",count);
    return ok;
}
int main(int argc,char** argv)
{
    if (argc!=4) { fprintf(stderr,"usage: token_probe MODEL UTF8_FILE OUTPUT_U32\n"); return 2; }
    char diagnostic[512]={0};
    MiciusTokenizer* t=Micius_OpenTokenizerV1(argv[1],diagnostic,sizeof(diagnostic));
    if (!t) { fprintf(stderr,"%s\n",diagnostic); return 1; }
    int ok=0;
    FILE* f=fopen(argv[2],"rb");
    if (f && fseek(f,0,SEEK_END)==0) {
        const long length=ftell(f);
        if (length>=0 && length<=16*1024*1024 && fseek(f,0,SEEK_SET)==0) {
            const size_t size=(size_t)length;
            char* text=(char*)malloc(size+1);
            if (text && fread(text,1,size,f)==size) {
                FILE* out=fopen(argv[3],"wb");
                if (out) { ok=check(t,text,size,out); if (fclose(out)!=0) ok=0; }
            }
            free(text);
        }
    }
    if (f) fclose(f);
    Micius_CloseTokenizerV1(t);
    return ok ? 0 : 1;
}
