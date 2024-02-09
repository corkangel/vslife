
#include "board.h"
#include "renderer.h"

#include <cuda_runtime.h>

#include "kernel.cuh"

int main()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1600, 1600, "Life", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    const unsigned int gridSize = 2000;

    Renderer renderer(gridSize);
    renderer.Initialize();

    CudaNeighborsBoard board(gridSize);

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

        renderer.HandleKeyInput(window);
    }

    renderer.Cleanup();
    glfwTerminate();

    return 0;
}