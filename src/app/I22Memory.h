#pragma once
#if defined(MICIUS_I6_BENCHMARK)
#include <windows.h>
#include <psapi.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace micius::i22 {
inline bool Observing() { const char* s=std::getenv("MICIUS_I22_MEMORY_FILE");return s && *s; }
inline std::string Quote(const char* text) {
    std::string out="\"";
    for(const unsigned char* p=reinterpret_cast<const unsigned char*>(text);*p;++p) {
        if(*p=='"' || *p=='\\') {out+='\\';out+=static_cast<char>(*p);}
        else if(*p<32) {const char* hex="0123456789abcdef";out+="\\u00";out+=hex[*p>>4];out+=hex[*p&15];}
        else out+=static_cast<char>(*p);
    }
    return out+'"';
}
inline void Memory(const char* phase,const std::string& extra="") {
    const char* file=std::getenv("MICIUS_I22_MEMORY_FILE");if(!file || !*file)return;
    PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof(m);DWORD handles=0;
    if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m)) ||
       !GetProcessHandleCount(GetCurrentProcess(),&handles))throw std::runtime_error("I22 process counters unavailable");
    // These are virtual-region extents. They MUST NOT be labelled resident bytes
    // or treated as private commit (copy-on-write pages keep their mapping type).
    uint64_t priv=0,mapped=0,image=0,reserved=0,regions=0;
    SYSTEM_INFO system{};GetSystemInfo(&system);uintptr_t addr=0;
    while(addr<=reinterpret_cast<uintptr_t>(system.lpMaximumApplicationAddress)) {
        MEMORY_BASIC_INFORMATION r{};
        if(!VirtualQuery(reinterpret_cast<void*>(addr),&r,sizeof(r))) {
            if(GetLastError()==ERROR_INVALID_PARAMETER)break;
            throw std::runtime_error("I22 region enumeration failed");
        }
        if(r.State==MEM_COMMIT) {
            if(r.Type==MEM_PRIVATE)priv+=r.RegionSize;
            else if(r.Type==MEM_MAPPED)mapped+=r.RegionSize;
            else if(r.Type==MEM_IMAGE)image+=r.RegionSize;
            ++regions;
        } else if(r.State==MEM_RESERVE)reserved+=r.RegionSize;
        const auto next=reinterpret_cast<uintptr_t>(r.BaseAddress)+r.RegionSize;
        if(next<=addr)throw std::runtime_error("I22 region address overflow");
        addr=next;
    }
    std::ofstream out(file,std::ios::binary|std::ios::app);
    out<<"{\"phase\":"<<Quote(phase)<<",\"pid\":"<<GetCurrentProcessId()
       <<",\"private_commit_bytes\":"<<m.PrivateUsage<<",\"peak_private_commit_bytes\":"<<m.PeakPagefileUsage
       <<",\"working_set_bytes\":"<<m.WorkingSetSize<<",\"peak_working_set_bytes\":"<<m.PeakWorkingSetSize
       <<",\"handles\":"<<handles<<",\"virtual_committed_private_bytes\":"<<priv
       <<",\"virtual_committed_mapped_bytes\":"<<mapped<<",\"virtual_committed_image_bytes\":"<<image
       <<",\"virtual_reserved_bytes\":"<<reserved<<",\"committed_regions\":"<<regions<<extra<<"}\n";
    out.close();if(!out)throw std::runtime_error("I22 memory evidence write failed");
}
}
#else
namespace micius::i22 {
inline void Memory(const char*) noexcept {}
}
#endif
