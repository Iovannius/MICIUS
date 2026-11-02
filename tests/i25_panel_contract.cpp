#include "PanelGEMV.h"
#include <array>
#include <bit>
#include <cmath>
#include <cfenv>
#include <cerrno>
#include <cstring>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif
using namespace micius::i23;
using namespace micius::internal;
static size_t cases=0,dots=0,fallbacks=0;
static void Need(bool ok,const char* s){if(!ok){std::cerr<<s<<" case="<<cases<<" dot="<<dots<<'\n';std::exit(2);}}
static uint32_t Next(uint32_t& s){return s=s*1664525U+1013904223U;}
static void DotCases(){
    std::array<int8_t,32>a;std::array<int8_t,128>w,expanded;std::array<QuantDescriptor,4>s;
    for(int av=-127;av<=127;++av)for(int wv=-128;wv<=127;++wv){
        a.fill(int8_t(av));
        for(size_t r=0;r<4;++r)for(size_t k=0;k<32;++k)
            w[r*32+k]=int8_t(r==0?wv:r==1?(k%2?-128:127):r==2?(k%2?wv:0):127-int(k)*8);
        for(size_t j=0;j<4;++j)for(size_t h=0;h<2;++h)for(size_t r=0;r<4;++r)for(size_t c=0;c<4;++c)
            expanded[j*32+h*16+r*4+c]=a[j*8+h*4+c];
        Matrix m;Need(m.Pack(w,s,4,32)==PackStatus::OK,"dot pack");
        int32_t got[4];_mm_storeu_si128(reinterpret_cast<__m128i*>(got),Dot(expanded.data(),m.weights()));
        for(size_t r=0;r<4;++r){int64_t want=0;for(size_t k=0;k<32;++k)want+=int64_t(a[k])*w[r*32+k];Need(want==got[r],"INT64 dot mismatch");}
        ++dots;
    }
    for(size_t k=0;k<32;++k){
        a.fill(0);a[k]=-127;for(size_t i=0;i<128;++i)w[i]=int8_t(int(i)-128);
        for(size_t j=0;j<4;++j)for(size_t h=0;h<2;++h)for(size_t r=0;r<4;++r)for(size_t c=0;c<4;++c)
            expanded[j*32+h*16+r*4+c]=a[j*8+h*4+c];
        Matrix m;Need(m.Pack(w,s,4,32)==PackStatus::OK,"lane pack");int32_t got[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(got),Dot(expanded.data(),m.weights()));
        for(size_t r=0;r<4;++r)Need(got[r]==int(a[k])*w[r*32+k],"lane order");++dots;
    }
}
static void GemvCase(uint32_t n,uint32_t rows,int pattern,unsigned csr){
    uint32_t seed=n*173+rows*111+uint32_t(pattern);const size_t nb=n/32;
    std::vector<int8_t>w(size_t(rows)*n),q(n),e(size_t(n)*4+2,71);
    std::vector<QuantDescriptor>s(rows*nb);std::vector<double>sc(nb);
    std::vector<float>input(n),old(rows+2,77),out=old,oracle(rows);
    for(auto&x:w)x=int8_t(int(Next(seed)%256)-128);
    for(auto&x:input){float v=float(int(Next(seed)%255)-127);
        x=pattern==0?0:pattern==1?v:pattern==2?v*std::numeric_limits<float>::denorm_min():pattern==3?v*(std::numeric_limits<float>::max()/128):v/19;}
    for(auto&x:s){float v=float(int(Next(seed)%31)-15);x=QuantDescriptor(std::ldexp(v,pattern==3?-40:-12));}
    if(pattern==5){for(auto& x:input)x=127;for(auto& x:w)x=127;for(auto&x:s)x=QuantDescriptor(std::numeric_limits<float>::max());}
    if(pattern==6){for(size_t i=0;i<s.size();++i)s[i]=QuantDescriptor(i%2?-0.0f:std::numeric_limits<float>::denorm_min());}
    const auto wb=w;const auto sb=s;const auto ib=input;
    Matrix m;Need(m.Pack(w,s,rows,n)==PackStatus::OK,"gemv pack");
    Need(m.payload()==w.size()+s.size()*4,"compact payload");
    Need(w==wb && std::memcmp(s.data(),sb.data(),s.size()*sizeof(s[0]))==0,"pack source mutation");
    for(size_t r=0;r<rows;++r)for(size_t b=0;b<nb;++b)
        Need(std::bit_cast<uint32_t>(m.scales()[(r/4*nb+b)*4+r%4])==std::bit_cast<uint32_t>(s[r*nb+b].scale),"scale bits");
    Input x;_mm_setcsr(0x1f80U);Need(PrepareW8A8Input(input,q,sc,x)==Status::OK,"prepare");
    Expanded ex;Need(Expand(x,{e.data()+1,size_t(n)*4},ex)==Status::OK,"expand");
    for(size_t b=0;b<nb;++b)for(size_t j=0;j<4;++j)for(size_t h=0;h<2;++h)for(size_t r=0;r<4;++r)for(size_t c=0;c<4;++c)
        Need(e[1+b*128+j*32+h*16+r*4+c]==q[b*32+j*8+h*4+c],"expanded lane");
    const auto qbefore=q,ebefore=e;const auto scbefore=sc;
    _mm_setcsr(csr);errno=17;const auto s0=GEMV_W8A8Rows(w.data(),s.data(),x,old.data()+1,rows,0,rows,true,false);const auto f0=_mm_getcsr();const auto err0=errno;
    _mm_setcsr(csr);errno=17;const auto s1=Consume(m,ex,{out.data()+1,rows});const auto f1=_mm_getcsr();
    Need(s0==s1,"status mismatch");Need(f0==f1 && errno==err0,"floating flags/errno mismatch");
    Need(std::memcmp(out.data(),old.data(),out.size()*4)==0,"output/partial failure/guard mismatch");
    Need(q==qbefore && e==ebefore && sc==scbefore && input==ib,"consumer input mutated");
    Need(e.front()==71 && e.back()==71,"expansion guard");
    // Independent scalar INT64 dot and chronological double postprocessing.
    _mm_setcsr(csr);
    if(s0==Status::OK){for(size_t r=0;r<rows;++r){double sum=0;for(size_t b=0;b<nb;++b){int64_t d=0;for(size_t k=0;k<32;++k)d+=int64_t(w[r*n+b*32+k])*q[b*32+k];
        const double v=double(d)*double(s[r*nb+b].scale);sum+=v*sc[b];}oracle[r]=float(sum);}
        Need(std::memcmp(oracle.data(),out.data()+1,rows*4)==0,"independent postprocessing mismatch");}
    // Ownership: poison and release all sources, then repeat from owned panel.
    if(s0==Status::OK){std::fill(w.begin(),w.end(),0);std::vector<int8_t>().swap(w);std::vector<QuantDescriptor>().swap(s);
        _mm_setcsr(csr);Need(Consume(m,ex,{out.data()+1,rows})==Status::OK,"owned repeat");Need(std::memcmp(out.data(),old.data(),out.size()*4)==0,"source lifetime dependence");}
    _mm_setcsr(0x1f80U);++cases;
}
static void Guards(){
    std::vector<int8_t>w(128,1),q(32),e(128);std::vector<QuantDescriptor>s(4);std::vector<float>x(32,1),out(4,91);std::vector<double>d(1);
    Matrix m;Need(m.Pack(w,s,4,32)==PackStatus::OK,"initial");const auto original=m.weights();
    s[0].zero_point=1;Need(m.Pack(w,s,4,32)==PackStatus::INVALID && m.weights()==original,"invalid pack transaction");s[0].zero_point=0;
    s[1].padding[1]=1;Need(m.Pack(w,s,4,32)==PackStatus::INVALID && m.weights()==original,"padding");s[1].padding[1]=0;
    s[0].scale=std::numeric_limits<float>::quiet_NaN();Need(m.Pack(w,s,4,32)==PackStatus::INVALID && m.weights()==original,"NaN pack");s[0].scale=1;
    Need(m.Pack(w,s,4,32,143)==PackStatus::RESOURCE && m.weights()==original,"budget transaction");
    Need(m.Pack({w.data(),127},s,4,32)==PackStatus::INVALID && m.weights()==original,"short source");
    Input in;Need(PrepareW8A8Input(x,q,d,in)==Status::OK,"guard prepare");Expanded ex;
    Need(Expand(in,e,ex)==Status::OK,"guard expand");
    Need(Expand(in,{e.data(),127},ex)==Status::INVALID_ARGUMENT && ex.cols()==0,"failed view invalidation");
    Need(Consume(m,ex,out)==Status::INVALID_ARGUMENT && out[0]==91,"invalid view write");
    Need(Expand(in,e,ex)==Status::OK,"restore view");Need(Consume(m,ex,{out.data(),3})==Status::INVALID_ARGUMENT && out[0]==91,"short output");
    for(unsigned csr:{0x3f80U,0x5f80U,0x7f80U,0x9f80U,0x1fc0U}){_mm_setcsr(csr);Need(Consume(m,ex,out)==Status::INVALID_ARGUMENT && _mm_getcsr()==csr && out[0]==91,"unsupported FP");}
    _mm_setcsr(0x1f80U);
    Matrix moved(std::move(m));Need(m.rows()==0 && Consume(m,ex,out)==Status::INVALID_ARGUMENT,"moved-from owner");
    m=std::move(moved);Need(moved.rows()==0 && m.weights()==original,"move ownership restore");
    for(uint32_t n:{1U,31U,32U,33U,64U})for(uint32_t rows:{1U,3U,5U}){
        std::vector<int8_t>ww(size_t(rows)*n,-128),qq(n);std::vector<QuantDescriptor>ss(size_t(rows)*((n+31)/32));
        Need(m.Pack(ww,ss,rows,n)==PackStatus::UNSUPPORTED && m.weights()==original,"unsupported changed owner");
        std::vector<float>xx(n,127),a(rows),b(rows);std::vector<double>dd((n+31)/32);Input v;
        Need(PrepareW8A8Input(xx,qq,dd,v)==Status::OK,"fallback prepare");
        Need(GEMV_W8A8Rows(ww.data(),ss.data(),v,a.data(),rows,0,rows,true,false)==Status::OK,"old fallback");
        Need(GEMV_W8A8Rows(ww.data(),ss.data(),v,b.data(),rows,0,rows,true,true)==Status::OK && a==b,"fallback independent oracle");++fallbacks;
    }
#ifdef _WIN32
    // End-of-page expansion buffers exercise all supported short block lengths.
    auto* mem=static_cast<int8_t*>(VirtualAlloc(nullptr,8192,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));Need(mem!=nullptr,"guard allocation");DWORD prev=0;
    Need(VirtualProtect(mem+4096,4096,PAGE_NOACCESS,&prev)!=0,"guard protection");
    for(uint32_t n:{32U,64U,96U,128U}){std::vector<float>xx(n,7),yy(4);std::vector<int8_t>qq(n),ww(4*n,11);std::vector<double>dd(n/32);std::vector<QuantDescriptor>ss(4*n/32);Input v;Expanded ev;Matrix mm;
        Need(PrepareW8A8Input(xx,qq,dd,v)==Status::OK && mm.Pack(ww,ss,4,n)==PackStatus::OK,"guard prepare 2");
        Need(Expand(v,{mem+4096-4*n,4*n},ev)==Status::OK && Consume(mm,ev,yy)==Status::OK,"guard end access");}
    Need(VirtualFree(mem,0,MEM_RELEASE)!=0,"guard free");
#endif
}
int main(){const auto saved=_mm_getcsr();_mm_setcsr(0x1f80U);DotCases();
    for(uint32_t n:{32U,64U,96U,960U,2560U})for(uint32_t rows:{4U,8U,12U,32U})for(int p=0;p<7;++p)for(unsigned csr:{0x1f80U,0x1fa0U,0x1fbfU})GemvCase(n,rows,p,csr);
    Guards();_mm_setcsr(saved);std::cout<<"{\"status\":\"PASS\",\"dot_cases\":"<<dots<<",\"gemv_cases\":"<<cases<<",\"fallback_cases\":"<<fallbacks<<",\"scope\":\"standalone panel contract only, no model integration or performance\"}\n";}
