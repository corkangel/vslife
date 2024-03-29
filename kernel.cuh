
//Simple
void update_board(const unsigned char* cells, unsigned char* new_cells, const unsigned int gridSize);

// OnGpu
void board_init(unsigned char* initialCells, const unsigned int boardSize);
void board_update(char* dirty);
void board_destroy();

// Neighbors
void neighbors_init(unsigned char* initialCells, const unsigned int boardSize);
void neighbors_update(char* dirty);
void neighbors_destroy();


// Neighbors Interop
void interop_init(unsigned char* initialCells, const unsigned int boardSize);
void interop_update(float* colorsDevicePtr);
void interop_reupload(unsigned char* newCells, float* colorsDevicePtr);
void interop_destroy();