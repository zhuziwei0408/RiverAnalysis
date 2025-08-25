#include <iostream>
#include "AnalysisManager.h"

const char* config_path = "./config/Configlist.config";

int main(int argc, char** argv)
{
    AnalysisManager manager;
    if (manager.LoadConfig(argv[0], config_path) != 0) {
        std::cout << "load config failed." << std::endl;
        return -1;
    }
    manager.run();
    return 0;
}

