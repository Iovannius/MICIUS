#include "AI/I2Diagnostic.h"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: micius_i2_replay INPUT.i2r OUTPUT.json\n";
        return 2;
    }
    try {
        micius::i2::Replay(argv[1],argv[2]);
        std::cout << "status=PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "status=FAILED\nerror=" << e.what() << '\n';
        return 1;
    }
}
