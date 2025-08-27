#pragma once
#include <string>

class Infer2DOD {
public:
    Infer2DOD(const std::string& model_path);
    ~Infer2DOD();
    void infer(const std::string& image_path);
private:
    // Add private members as needed
};