#include "AI/LoggerCore.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace micius::log;
namespace {
void need(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
}
int main()
{
    try {
        auto& logger=AILoggerCore::GetInstance();
        logger.Stop();
        logger.CommitLog(LogLevel::INFO,__FILE__,10,0,0,"disabled");
        need(!logger.Start(-1,""),"empty path");
        auto dir=std::filesystem::temp_directory_path()/("micius-log-"+std::to_string(AILoggerCore::GetFastNanoTime()));
        need(std::filesystem::create_directory(dir),"temp directory");
        const auto log=dir/"events.log";
        need(!logger.Start(-1,(dir/"absent"/"events.log").string().c_str()),"bad path");
        need(logger.Start(-1,log.string().c_str()),"retry open");
        need(!logger.Start(-1,log.string().c_str()),"duplicate start");
        for (int i=0;i<32;++i) logger.CommitLog(LogLevel::WARN,"fixture",42,7,3,"mix=%d/%f/%u %%",
                                               -7,0.5,123u);
        logger.Stop();
        logger.CommitLog(LogLevel::INFO,"fixture",1,0,0,"between sessions");
        need(logger.Start(-1,log.string().c_str()),"reopen");
        logger.CommitLog(LogLevel::ERR,"fixture",1,0,0,"second"); logger.Stop();
        std::ifstream in(log);
        std::string line;size_t count=0;
        while (std::getline(in,line)) {
            if (count<32) need(line.ends_with("[WRN][E:7][T:3][fixture:42] mix=-7/0.500000/123 %"),"typed event");
            else need(line.ends_with("[ERR][E:0][T:0][fixture:1] second"),"session isolation");
            ++count;
        }
        need(count==33,"lost or stale records"); in.close();std::filesystem::remove(log);std::filesystem::remove(dir);
        std::cout<<"Logger fail-closed open, mixed fields, close drain and session separation: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
