
#include "utils.h"


void DrawGrid(const char* label, uint32 sz, const std::vector<uint32>& data)
{
	printf("------------------ %s ------------------\n", label);
	for (uint32 y = 0; y != sz; y++)
	{
		for (uint32 x = 0; x < sz; x++)
		{
			printf("%d ", data[y * sz + x]);
		}
		printf("\n");
	}
}

void DrawGrid(const char* label, uint32 sz, const std::vector<uint8>& data)
{
	printf("------------------ %s ------------------\n", label);
	for (uint32 y = 0; y != sz; y++)
	{
		for (uint32 x = 0; x < sz; x++)
		{
			printf("%d ", data[y * sz + x]);
		}
		printf("\n");
	}
}


void DrawGrid(const char* label, uint32 sz, const std::vector<int8>& data)
{
	printf("------------------ %s ------------------\n", label);
	for (uint32 y = 0; y != sz; y++)
	{
		for (uint32 x = 0; x < sz; x++)
		{
			printf("%+d ", data[y * sz + x]);
		}
		printf("\n");
	}
}
