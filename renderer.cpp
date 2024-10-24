#include "renderer.h"

#include <glm/gtc/type_ptr.hpp>

const char* vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aColor;

uniform mat4 transform;

out float ourColor; // Output to the fragment shader

void main()
{
    gl_Position = transform * vec4(aPos, 1.0);
    ourColor = aColor; // Pass the color to the fragment shader
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
in float ourColor; // Input from the vertex shader

out vec4 FragColor;

void main()
{
    FragColor = vec4(ourColor, ourColor, ourColor, 1.0f);
}
)glsl";


void Renderer::HandleKeyInput(GLFWwindow* window)
{
    // Check if the '+' key is pressed
    if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        // Zoom in by scaling the camera transform
        cameraTransform = glm::scale(cameraTransform, glm::vec3(1.01f, 1.01f, 1.01f));
    }

    // Check if the '-' key is pressed
    if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        // Zoom out by scaling the camera transform
        cameraTransform = glm::scale(cameraTransform, glm::vec3(0.99f, 0.99f, 0.99f));
    }

    // Check if the 'W' key is pressed
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        // Move the camera up
        cameraTransform = glm::translate(cameraTransform, glm::vec3(0.0f, 0.01f, 0.0f));
    }

    // Check if the 'A' key is pressed
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        // Move the camera left
        cameraTransform = glm::translate(cameraTransform, glm::vec3(-0.01f, 0.0f, 0.0f));
    }

    // Check if the 'S' key is pressed
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        // Move the camera down
        cameraTransform = glm::translate(cameraTransform, glm::vec3(0.0f, -0.01f, 0.0f));
    }

    // Check if the 'D' key is pressed
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        // Move the camera right
        cameraTransform = glm::translate(cameraTransform, glm::vec3(0.01f, 0.0f, 0.0f));
    }
}

void Renderer::ApplyCameraTransform()
{
    // Apply the camera transform to the shader program
    GLuint transformLoc = glGetUniformLocation(shaderProgram, "transform");
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(cameraTransform));
}

GLuint Renderer::MakeShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource)
{
    // Compile the vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Compile the fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Delete the vertex and fragment shaders as they're no longer needed
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void Renderer::Initialize()
{
    shaderProgram = MakeShaderProgram(vertexShaderSource, fragmentShaderSource);

    const float sz = 0.01f;
    const float half = gridSize * sz / 2;
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            // Generate vertices for each square (2 triangles)
            vertices.insert(vertices.end(), {
                float(i * sz) - half,      float(j * sz) - half,     0.0f,
                float(i * sz) - half + sz, float(j * sz) - half,     0.0f,
                float(i * sz) - half + sz, float(j * sz) - half + sz, 0.0f,
                float(i * sz) - half,      float(j * sz) - half + sz, 0.0f
                });

            // Generate indices for each square
            unsigned int start = (i * gridSize + j) * 4;
            indices.insert(indices.end(), {
                start, start + 1, start + 2,
                start, start + 2, start + 3
                });

            // Generate color for each square (all vertices of a square have the same color)
            colors.insert(colors.end(), 4, 0.0f);
        }
    }

    // Bind the VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBOverts);
    glBindBuffer(GL_ARRAY_BUFFER, VBOverts);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &VBOcolors);
    glBindBuffer(GL_ARRAY_BUFFER, VBOcolors);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

    // // Unbind the VBO and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Renderer::UpdateColorBuffer()
{
    glBindBuffer(GL_ARRAY_BUFFER, VBOcolors);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), &colors[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::Draw()
{
    ApplyCameraTransform();

    if (updateColorBuffer) {
        UpdateColorBuffer();
    }
    
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, GLsizei(indices.size()), GL_UNSIGNED_INT, 0);
}

void Renderer::Cleanup()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBOverts);
    glDeleteBuffers(1, &VBOcolors);
    glDeleteBuffers(1, &EBO);
}

glm::vec2 Renderer::GetCameraFocus() {
    glm::vec3 cameraPosition = glm::vec3(cameraTransform[3]);
    return glm::vec2(cameraPosition.x, cameraPosition.y);
}
