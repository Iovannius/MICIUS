#include "Storage.h"
#include "gguf/NativePath.h"
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <random>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace micius::session {
namespace fs=std::filesystem;
using micius::gguf::NativePath;
std::string Utf8(const fs::path& path) { auto s=path.u8string();return {s.begin(),s.end()}; }
void PlainDirectory(const fs::path& path) {
    // Do not traverse a redirected model/session/kv directory.
    for(auto p=fs::absolute(path).lexically_normal();!p.empty();p=p.parent_path()) {
        auto status=fs::symlink_status(NativePath(p));
        if(!fs::is_directory(status) || fs::is_symlink(status))throw std::runtime_error("Expected a plain session directory");
#ifdef _WIN32
        auto attr=GetFileAttributesW(NativePath(p).c_str());
        if(attr==INVALID_FILE_ATTRIBUTES || (attr&FILE_ATTRIBUTE_REPARSE_POINT))throw std::runtime_error("Redirected session directory");
#endif
        if(p==p.root_path())break;
    }
}
std::vector<uint8_t> ReadFile(const fs::path& path,size_t limit) {
    auto p=NativePath(path); auto s=fs::symlink_status(p);
    if(!fs::is_regular_file(s) || fs::is_symlink(s))throw std::runtime_error("Expected an ordinary session file: "+Utf8(path));
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    const auto end=f.tellg();
    if(!f || end<0 || static_cast<uint64_t>(end)>limit)throw std::runtime_error("Cannot read bounded session file");
    std::vector<uint8_t> bytes(static_cast<size_t>(end));f.seekg(0);
    if(!bytes.empty())f.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if(!f || f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Session file changed/truncated while reading");
    return bytes;
}
Lock::Lock(const fs::path& path) {
    auto p=NativePath(path);
    if(fs::is_symlink(fs::symlink_status(p)))throw std::runtime_error("Redirected session lock");
#ifdef _WIN32
    auto h=CreateFileW(p.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("Session is already open or cannot be locked");
    handle=reinterpret_cast<intptr_t>(h);
#else
    int fd=open(p.c_str(),O_RDWR|O_CREAT|O_NOFOLLOW,0600);
    if(fd<0)throw std::runtime_error("Cannot open session lock");
    if(flock(fd,LOCK_EX|LOCK_NB)) { close(fd);throw std::runtime_error("Session is already open"); }
    handle=fd;
#endif
}
Lock::~Lock() {
    if(handle==-1)return;
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
    flock(static_cast<int>(handle),LOCK_UN);close(static_cast<int>(handle));
#endif
}
void AtomicWrite(const fs::path& destination,std::span<const uint8_t> bytes) {
    PlainDirectory(destination.parent_path());
    if(fs::is_symlink(fs::symlink_status(NativePath(destination))))throw std::runtime_error("Redirected session file");
    const auto native=NativePath(destination);
    std::random_device random;
    fs::path temp;
#ifdef _WIN32
    HANDLE h=INVALID_HANDLE_VALUE;
    for(int i=0;i<16 && h==INVALID_HANDLE_VALUE;++i) {
        temp=NativePath(destination.parent_path()/(".micius-"+std::to_string(random())+".tmp"));
        h=CreateFileW(temp.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(h==INVALID_HANDLE_VALUE && GetLastError()!=ERROR_FILE_EXISTS)throw std::runtime_error("Cannot create snapshot temporary file");
    }
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot reserve snapshot temporary file");
    try {
        size_t done=0;
        while(done<bytes.size()) {
            DWORD written=0,chunk=static_cast<DWORD>(std::min<size_t>(bytes.size()-done,1u<<20));
            if(!WriteFile(h,bytes.data()+done,chunk,&written,nullptr) || written!=chunk)throw std::runtime_error("Snapshot write failed");
            done+=written;
        }
        if(!FlushFileBuffers(h))throw std::runtime_error("Snapshot flush failed");
        CloseHandle(h);h=INVALID_HANDLE_VALUE;
        if(!MoveFileExW(temp.c_str(),native.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Snapshot atomic replacement failed");
    } catch(...) { if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);DeleteFileW(temp.c_str());throw; }
#else
    int fd=-1;
    for(int i=0;i<16 && fd<0;++i) {
        temp=destination.parent_path()/(".micius-"+std::to_string(random())+".tmp");
        fd=open(temp.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
        if(fd<0 && errno!=EEXIST)throw std::runtime_error("Cannot create snapshot temporary file");
    }
    if(fd<0)throw std::runtime_error("Cannot reserve snapshot temporary file");
    try {
        size_t done=0;
        while(done<bytes.size()) { auto n=write(fd,bytes.data()+done,std::min<size_t>(bytes.size()-done,1u<<20));if(n<0 && errno==EINTR)continue;if(n<=0)throw std::runtime_error("Snapshot write failed");done+=static_cast<size_t>(n); }
        if(fsync(fd))throw std::runtime_error("Snapshot flush failed");
        close(fd);fd=-1;
        if(rename(temp.c_str(),native.c_str()))throw std::runtime_error("Snapshot atomic replacement failed");
        int dir=open(destination.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
        if(dir>=0) { int result=fsync(dir);close(dir);if(result)throw std::runtime_error("Snapshot directory sync failed"); }
        else throw std::runtime_error("Cannot sync snapshot directory");
    } catch(...) { if(fd>=0)close(fd);unlink(temp.c_str());throw; }
#endif
}
void AtomicText(const fs::path& path,const std::string& text) {
    AtomicWrite(path,{reinterpret_cast<const uint8_t*>(text.data()),text.size()});
}
}
