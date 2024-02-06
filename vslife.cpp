
#include "board.h"
#include "renderer.h"

#include <cuda_runtime.h>

#include "kernel.cuh"


void cuda()
{
    // Initialize arrays A, B, and C.
    double A[3], B[3], C[3];

    // Populate arrays A and B.
    A[0] = 5; A[1] = 8; A[2] = 3;
    B[0] = 7; B[1] = 6; B[2] = 4;

    // Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA.
    kernel(A, B, C, 3);

    printf("C[0] = %f\n", C[0]);
    printf("C[1] = %f\n", C[1]);
    printf("C[2] = %f\n", C[2]);
}

int main()
{
    cuda();

    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1600, 1600, "Life", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    const unsigned int gridSize = 2000;

    Renderer renderer(gridSize);
    renderer.Initialize();

    Board board(gridSize);

    const unsigned int numFrames = 100;
    int frameCount = 0;
    std::vector<float> frameTimes(numFrames, 0.0f);

    while (!glfwWindowShouldClose(window))
    {
        auto start = std::chrono::high_resolution_clock::now();

        board.Update();

        board.Draw(renderer.colors);

        frameTimes[frameCount % numFrames] = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(std::chrono::high_resolution_clock::now() - start).count();
        if (frameCount++ % 50 == 0)
        {
            const float avg = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) / numFrames;
            printf("Time: %u %.2f FPS\n", frameCount, 1000 / avg);
        }

        glClear(GL_COLOR_BUFFER_BIT);
        renderer.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();

        // handle keyboard event
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
    }

    renderer.Cleanup();
    glfwTerminate();

    return 0;
}