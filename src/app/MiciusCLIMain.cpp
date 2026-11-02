#include "MiciusShell.h"
#include <exception>
#include <iostream>
#include <string>
#include <vector>

void PrintBanner()
{
    std::cout << "========================================================\n";
    std::cout << " MICIUS Shell v" MICIUS_VERSION_STRING "\n";
    std::cout << "========================================================\n\n";
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "MICIUS " MICIUS_VERSION_STRING "\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "micius_cli [--version | --model-benchmark --help]\n"
                     "Start without arguments for the interactive shell.\n";
        return 0;
    }
    PrintBanner();

    //实例化外壳
    MiciusShell shell;

    if (argc > 1) {
        if (std::string(argv[1]) != "--model-benchmark") {
            std::cerr << "Unknown non-interactive mode: " << argv[1] << '\n';
            return 2;
        }
        std::vector<std::string> args;
        args.reserve(static_cast<size_t>(argc - 2));
        for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);
        try {
            return shell.RunModelBenchmark(args);
        } catch (const std::exception& error) {
            std::cerr << "status=FAIL\nerror=model benchmark aborted by exception: "
                      << error.what() << '\n';
            return 1;
        } catch (...) {
            std::cerr << "status=FAIL\nerror=model benchmark aborted by unknown exception\n";
            return 1;
        }
    }
    
    //启动REPL解释器
    shell.RunREPL();

    return 0;
}
