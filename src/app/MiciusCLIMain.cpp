#include "CliArtwork.h"
#include "MiciusShell.h"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
bool LogoUsesColor()
{
    const char* no_color = std::getenv("NO_COLOR");
    const char* term = std::getenv("TERM");
    if ((no_color && *no_color) || (term && std::string_view(term) == "dumb"))
        return false;
#ifdef _WIN32
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(output, &mode))
        return (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;

    // A native program in Mintty can receive a named PTY pipe for stdout.
    // An ordinary pipeline can inherit the very same TERM value.
    struct { DWORD length; WCHAR text[256]; } pipe{};
    if (GetFileType(output) != FILE_TYPE_PIPE ||
        !GetFileInformationByHandleEx(output, FileNameInfo, &pipe, sizeof(pipe)) ||
        pipe.length > sizeof(pipe.text))
        return false;
    std::wstring_view name(pipe.text, pipe.length / sizeof(WCHAR));
    const auto separator = name.rfind(L'\\');
    if (separator != name.npos) name.remove_prefix(separator + 1);
    return (name.starts_with(L"msys-") || name.starts_with(L"cygwin-")) &&
           name.find(L"-pty") != name.npos && name.find(L"-to-master") != name.npos;
#else
    return isatty(STDOUT_FILENO) == 1;
#endif
}

void PrintBanner()
{
    using micius::cli::artwork;
    constexpr std::string_view placeholder = "XXXX";
    constexpr auto marker = artwork.find(placeholder);
    static_assert(marker != artwork.npos &&
                  artwork.find(placeholder, marker + placeholder.size()) == artwork.npos,
                  "MICIUS.txt must contain exactly one XXXX color marker");
    constexpr std::string_view block = "\xe2\x96\x88"; // █ in UTF-8
    std::cout << artwork.substr(0, marker);
    if (LogoUsesColor()) {
        for (const char* color : {"\x1b[31m", "\x1b[33m", "\x1b[34m", "\x1b[32m"})
            std::cout << color << block;
        std::cout << "\x1b[39m";
    } else {
        std::cout << "\xe2\x96\x91\xe2\x96\x92\xe2\x96\x93" << block; // ░▒▓█
    }
    std::cout << artwork.substr(marker + placeholder.size());
    if (!artwork.ends_with('\n')) std::cout.put('\n');
    std::cout << " MICIUS Shell v" MICIUS_VERSION_STRING "\n\n";
}
} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "MICIUS " MICIUS_VERSION_STRING "\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "micius_cli [--version | --model-benchmark --help]\n"
#ifdef MICIUS_HAS_TEXT
                     "micius_cli --chat MODEL [--context N] [--max-new N] [--session ID]\n"
#endif
                     "Start without arguments for the interactive shell.\n";
        return 0;
    }
    //实例化外壳
    MiciusShell shell;
    PrintBanner(); // The shell first enables UTF-8 and virtual-terminal output on Windows.

    if (argc > 1) {
#ifdef MICIUS_HAS_TEXT
        if (std::string(argv[1])=="--chat") {
            try { return shell.RunChat(std::vector<std::string>(argv+2,argv+argc)); }
            catch (const std::exception& e) { std::cerr<<"[Chat] "<<e.what()<<'\n'; return 1; }
        }
#endif
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
