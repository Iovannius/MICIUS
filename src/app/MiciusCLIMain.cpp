#include "MiciusShell.h"
#include <iostream>

void PrintBanner()
{
    std::cout << "========================================================\n";
    std::cout << " MICIUS Shell v0.5.1\n";
    std::cout << "========================================================\n\n";
}

int main(int argc, char** argv)
{
    PrintBanner();

    //实例化外壳
    MiciusShell shell;
    
    //启动REPL解释器
    shell.RunREPL();

    return 0;
}