#include "quant/WeightFormat.h"
#include <fstream>
#include <iterator>
#include <vector>
#include <cstdlib>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=5)return 2;
    const auto type=static_cast<micius::quant::Type>(std::strtoul(argv[1],nullptr,10));
    const size_t n=std::strtoul(argv[2],nullptr,10);
    if(!n || n>1048576)return 2;
    std::ifstream in(argv[3],std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});
    std::vector<float> output(n);
    if(!micius::quant::Decode(type,bytes.data(),bytes.size(),output.data(),n))return 1;
    std::ofstream out(argv[4],std::ios::binary);
    out.write(reinterpret_cast<const char*>(output.data()),output.size()*4);
    return out.good()?0:2;
}
