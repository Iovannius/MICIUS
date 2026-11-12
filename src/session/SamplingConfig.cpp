#include "SamplingConfig.h"
#include <charconv>
#include <cmath>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace micius::session {
namespace {
std::string_view Trim(std::string_view s) {
    auto first=s.find_first_not_of(" \t\r"); if(first==s.npos)return {};
    return s.substr(first,s.find_last_not_of(" \t\r")-first+1);
}
template<class T> void Number(std::string_view value,T& output) {
    T n{};auto r=std::from_chars(value.data(),value.data()+value.size(),n);
    if(value.empty() || r.ec!=std::errc{} || r.ptr!=value.data()+value.size())
        throw std::invalid_argument("Invalid sampling configuration number");
    output=n;
}
}
Config ParseConfig(std::string_view text) {
    if(text.size()>16384)throw std::invalid_argument("Sampling configuration too large");
    Config c; std::set<std::string> keys; bool version=false;
    while(!text.empty()) {
        auto end=text.find('\n'); auto line=Trim(text.substr(0,end));
        text=end==text.npos ? std::string_view{} : text.substr(end+1);
        if(line.empty() || line.front()=='#')continue;
        auto equal=line.find('=');if(equal==line.npos)throw std::invalid_argument("Expected key=value in sampling.conf");
        auto key=Trim(line.substr(0,equal)), value=Trim(line.substr(equal+1));
        if(!keys.emplace(key).second)throw std::invalid_argument("Duplicate sampling key");
        if(key=="version") { uint32_t v=0;Number(value,v);if(v!=1)throw std::invalid_argument("Unsupported sampling version"); version=true; }
        else if(key=="temperature")Number(value,c.sample.temperature);
        else if(key=="top_p")Number(value,c.sample.top_p);
        else if(key=="top_k")Number(value,c.sample.top_k);
        else if(key=="repeat_penalty")Number(value,c.sample.repeat_penalty);
        else if(key=="repeat_last_n")Number(value,c.sample.repeat_last_n);
        else if(key=="seed")Number(value,c.seed);
        else if(key=="max_new_tokens")Number(value,c.max_new_tokens);
        else throw std::invalid_argument("Unknown sampling key: "+std::string(key));
    }
    if(!version || !c.max_new_tokens || !std::isfinite(c.sample.temperature) || c.sample.temperature<0 ||
       !std::isfinite(c.sample.top_p) || c.sample.top_p<=0 || c.sample.top_p>1 ||
       !std::isfinite(c.sample.repeat_penalty) || c.sample.repeat_penalty<=0)
        throw std::invalid_argument("Invalid sampling parameter range/version");
    return c;
}
std::string WriteConfig(const Config& c) {
    std::ostringstream out;out.imbue(std::locale::classic());out.precision(17);
    out<<"version=1\ntemperature="<<c.sample.temperature<<"\ntop_k="<<c.sample.top_k<<"\ntop_p="<<c.sample.top_p
       <<"\nrepeat_penalty="<<c.sample.repeat_penalty<<"\nrepeat_last_n="<<c.sample.repeat_last_n
       <<"\nseed="<<c.seed<<"\nmax_new_tokens="<<c.max_new_tokens<<'\n';return out.str();
}
}
