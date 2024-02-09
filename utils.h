
#pragma once

using uint32 = unsigned int;
using int32 = int;
using uint8 = unsigned char;
using int8 = char;	

#include <vector>

void DrawGrid(const char* label, uint32 sz, const std::vector<uint32>& data);
void DrawGrid(const char* label, uint32 sz, const std::vector<uint8>& data);
void DrawGrid(const char* label, uint32 sz, const std::vector<int8>& data);