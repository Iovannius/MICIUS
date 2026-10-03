// Reuse the unchanged runtime contract suite while D1 collection is active.
#include "AI/DecodeProfile.h"
#define main existing_runtime_contract_main
#include "runtime_contract_tests.cpp"
#undef main
int main() {
    micius::d1::Session session(true);
    if (!session.valid()) return 2;
    const int result=existing_runtime_contract_main();
    if (!session.stop()) return 3;
    return result;
}
