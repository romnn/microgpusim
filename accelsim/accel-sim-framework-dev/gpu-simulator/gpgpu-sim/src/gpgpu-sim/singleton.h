#pragma once
#include <iostream>
#include <mutex>

class Singleton {
public:
    static Singleton* Instance();

    static void mem_printf(const char *format, ...);
    FILE *get_mem_debug_file();
protected:
    Singleton() {
        this->_instance = nullptr;
        this->mem_debug_file = fopen("./accelsim_mem_debug_trace.txt", "w");
        printf("ROMAN: opened accelsim_mem_debug_trace.txt\n");
    }
    std::mutex mtx;
private:
    static Singleton* _instance;
    FILE *mem_debug_file;
};