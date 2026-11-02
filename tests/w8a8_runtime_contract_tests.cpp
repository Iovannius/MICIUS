// Reuse ABI/ownership/lifecycle checks unchanged. The F64 asymmetric numerical
// oracle is run in BASE only; I1 uses its independent integer oracle plus
// SCALAR/AVX2 full-model replay. Its F64 tolerance must not be relaxed.
#define main F64ContractSuiteNotInvoked
#include "runtime_contract_tests.cpp"
#undef main
int main() {
    TestConfigAndAbiValidation();
    TestReferenceLifecycleAndFailClosedStep();
    TestDeterministicReferenceDecodeAndContextOwnership();
    TestLegacyPrefixCreation();
    TestDraftProposalBoundedReservation();
    std::cout<<"I1 ABI/lifecycle checks passed; F64 asymmetric oracle belongs to BASE.\n";
}
