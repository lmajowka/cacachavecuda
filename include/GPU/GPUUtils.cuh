#ifndef GPUUTILS_CUH
#define GPUUTILS_CUH

#include <string>
#include <sstream>
#include <iomanip>

// Função auxiliar para converter bytes para hex
inline std::string bytesToHex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for(size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

// Remove espaços de uma string
inline std::string removeSpaces(const std::string& str) {
    std::string result;
    for(char c : str) {
        if(c != ' ') {
            result += c;
        }
    }
    return result;
}

#endif // GPUUTILS_CUH
