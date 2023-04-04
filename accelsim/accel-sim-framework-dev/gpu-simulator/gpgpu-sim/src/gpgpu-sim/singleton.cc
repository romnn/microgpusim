#include "singleton.h";
#include <iostream>
#include <mutex>

Singleton* Singleton::_instance = nullptr;

void Singleton::mem_printf(const char *format, ...) {
    printf("ROMAN: printing to accelsim_mem_debug_trace.txt\n");
    Singleton *instance = Singleton::Instance();
    // lock instance
    instance->mtx.lock();
    printf("ROMAN: locked accelsim_mem_debug_trace.txt\n");
    fprintf(instance->mem_debug_file, format);
    fflush(instance->mem_debug_file);
    // unlock instance
    instance->mtx.unlock();
    printf("ROMAN: unlocked accelsim_mem_debug_trace.txt\n");
}

FILE *Singleton::get_mem_debug_file() {
    return this->mem_debug_file;
}

Singleton* Singleton::Instance() {
    if(_instance == nullptr) {
        _instance = new Singleton();
    }
    return _instance;
}