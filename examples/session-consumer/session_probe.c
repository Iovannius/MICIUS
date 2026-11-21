/* Public C SDK example and bounded resume check. Input lines are messages;
 * :reset and :save are commands. Snapshots are explicitly saved by the owner. */
#include "micius_session.h"
#include "micius_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <wchar.h>
#endif
/* The SDK returns UTF-8 directories, which can exceed MAX_PATH in a nested
 * validation checkout. Use the same wide, absolute path semantics for the
 * example's evidence/config files as the library uses for snapshots. */
static FILE* open_file(const char* name,const char* mode) {
#ifdef _WIN32
    int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,name,-1,NULL,0);
    if(!n)return NULL;
    wchar_t* wide=(wchar_t*)malloc((size_t)n*sizeof(wchar_t));
    if(!wide)return NULL;
    if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,name,-1,wide,n)) { free(wide);return NULL; }
    for(int i=0;i<n;++i)if(wide[i]==L'/')wide[i]=L'\\';
    DWORD size=GetFullPathNameW(wide,0,NULL,NULL);
    wchar_t* absolute=size?(wchar_t*)malloc((size_t)size*sizeof(wchar_t)):NULL;
    if(!absolute || !GetFullPathNameW(wide,size,absolute,NULL)) { free(wide);free(absolute);return NULL; }
    free(wide);
    wchar_t* path=(wchar_t*)malloc(((size_t)size+8)*sizeof(wchar_t));
    if(!path) { free(absolute);return NULL; }
    if(!wcsncmp(absolute,L"\\\\?\\",4))wcscpy(path,absolute);
    else if(!wcsncmp(absolute,L"\\\\",2)) { wcscpy(path,L"\\\\?\\UNC\\");wcscat(path,absolute+2); }
    else { wcscpy(path,L"\\\\?\\");wcscat(path,absolute); }
    FILE* result=_wfopen(path,mode[0]=='r'?L"rb":L"wb");
    free(path);free(absolute);return result;
#else
    return fopen(name,mode);
#endif
}
static FILE* answer;
static int emit(const char* s,size_t n,void* unused) { (void)unused;return fwrite(s,1,n,answer)==n; }
static FILE* output(const char* root,const char* name) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",root,name);
    return n<0 || (size_t)n>=sizeof(path)?NULL:open_file(path,"wb");
}
int main(int argc,char** argv) {
    if(argc!=7 && argc!=8) { fprintf(stderr,"MODEL CACHE_ROOT CONTEXT ID_OR_- INPUT_UTF8 OUTPUT_DIR [NEW_CONFIG]\n");return 2; }
    const MiciusGGUFParserV1* parsers[]={Micius_LlamaParserV1(),Micius_Qwen2ParserV1()
#ifdef LINK_QWEN3
        ,Micius_Qwen3ParserV1()
#endif
    };
    MiciusGGUFOpenOptionsV1 o={MICIUS_GGUF_FILE_ABI,sizeof(o),argv[1],argv[2],parsers,sizeof(parsers)/sizeof(parsers[0]),(uint32_t)strtoul(argv[3],0,10)};
    char error[1024]={0};int ok=0;
    MiciusFileModel* f=Micius_OpenGGUFV1(&o,error,sizeof(error));
    if(!f) { fprintf(stderr,"%s\n",error);return 1; }
    if(argc==8) {
        if(strcmp(argv[4],"-")) { fprintf(stderr,"NEW_CONFIG requires a new context\n");Micius_CloseGGUFV1(f);return 2; }
        FILE* src=open_file(argv[7],"rb");FILE* dst=output(Micius_FileContextDirectoryV1(f),"sampling.conf");
        int valid=src && dst;char bytes[16385];size_t n=src?fread(bytes,1,sizeof(bytes),src):0;
        if(n>16384 || (src && ferror(src)) || (valid && fwrite(bytes,1,n,dst)!=n))valid=0;
        if(src)fclose(src);if(dst && fclose(dst))valid=0;
        if(!valid) { fprintf(stderr,"Cannot initialize configuration\n");Micius_CloseGGUFV1(f);return 1; }
    }
    MiciusChatSession* chat=Micius_OpenChatV1(f,strcmp(argv[4],"-")?argv[4]:NULL,error,sizeof(error));
    if(!chat) { fprintf(stderr,"%s\n",error);Micius_CloseGGUFV1(f);return 1; }
    FILE* dir=output(argv[6],"directory.txt");
    if(dir) { fputs(Micius_ChatDirectoryV1(chat),dir);fclose(dir); }
    FILE* in=open_file(argv[5],"rb");FILE* events=output(argv[6],"events.txt");
    if(!in || !dir || !events)goto done;
    char line[8192];unsigned turn=0;
    while(fgets(line,sizeof(line),in)) {
        size_t n=strlen(line);while(n && (line[n-1]=='\r'||line[n-1]=='\n'))line[--n]=0;
        if(!n)continue;
        if(!strcmp(line,":reset")) {
            if(!Micius_ResetChatV1(chat,error,sizeof(error)))goto done;
        }else if(strcmp(line,":save")) {
            char name[80];snprintf(name,sizeof(name),"turn-%u.utf8",++turn);answer=output(argv[6],name);
            if(!answer)goto done;
            MiciusChatResultV1 r={MICIUS_SESSION_ABI_V1,sizeof(r),0,0,0};
            int valid=Micius_ChatReplyV1(chat,line,n,0,emit,NULL,&r,error,sizeof(error));
            int closed=fclose(answer);answer=NULL;
            if(!valid || closed || r.finish>2)goto done;
            fprintf(events,"%u %u %u %u\n",turn,r.finish,r.tokens,r.context_length);
        }
        if(!Micius_SaveChatV1(chat,error,sizeof(error)))goto done;
    }
    if(ferror(in) || !Micius_SaveChatV1(chat,error,sizeof(error)))goto done;
    {
        size_t n=0;MiciusContext* engine=Micius_FileEngineV1(f);
        if(Micius_ExportStateV1(engine,NULL,0,&n)!=0)goto done;
        void* raw=malloc(n);FILE* state=output(argv[6],"state.bin");
        int valid=raw && state && Micius_ExportStateV1(engine,raw,n,&n)==1 && fwrite(raw,1,n,state)==n;
        free(raw);if(state && fclose(state))valid=0;
        if(!valid)goto done;
    }
    ok=1;
done:
    if(in)fclose(in);if(events && fclose(events))ok=0;
    Micius_CloseChatV1(chat);Micius_CloseGGUFV1(f);
    if(!ok)fprintf(stderr,"session check failed: %s\n",error);
    else puts("SESSION_CHECK_PASS");
    return ok?0:1;
}
