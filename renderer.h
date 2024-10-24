#pragma once

#include <vector>
#include <numeric>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


class Renderer
{
public:
    Renderer(unsigned int sz) : gridSize(sz)
    {
        cameraTransform = glm::mat4(1.0f);
    }

    void Initialize();
    void Draw();
    void Cleanup();

    std::vector<GLfloat>& GetColors() { return colors; }
    unsigned int GetColorVBO() { return VBOcolors; }
    void SetUpdateColorBuffer(bool value) { updateColorBuffer = value; }

    void HandleKeyInput(GLFWwindow* window);
    glm::vec2 GetCameraFocus(); // Pbcca

protected:

    void ApplyCameraTransform();
    void UpdateColorBuffer();
    GLuint MakeShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource);

    unsigned int VAO;
    unsigned int EBO;
    unsigned int VBOverts, VBOcolors;
    GLuint shaderProgram;
    bool updateColorBuffer = true;

    const int gridSize;
    std::vector<GLfloat> vertices;
    std::vector<GLuint> indices;
    std::vector<GLfloat> colors;

    glm::mat4 cameraTransform;
};
