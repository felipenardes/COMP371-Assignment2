#define PLATFORM_OSX

#include <iostream>
#include <list>
#include <algorithm>

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <GLFW/glfw3.h> 

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <glm/common.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "OBJloader.h"

using namespace glm;
using namespace std;

GLuint loadTexture(const char *filename)
{
    // Load Textures with dimension data
    int width, height, nrChannels;
    unsigned char *data = stbi_load(filename, &width, &height, &nrChannels, 0);
    if (!data)
    {
        std::cerr << "Error::Texture could not load texture file: " << filename << std::endl;
        return 0;
    }

    // Create and bind textures
    GLuint textureId = 0;
    glGenTextures(1, &textureId);
    assert(textureId != 0);

    glBindTexture(GL_TEXTURE_2D, textureId);

    // Set filter parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload the texture to the GPU
    GLenum format = 0;
    if (nrChannels == 1)
        format = GL_RED;
    else if (nrChannels == 3)
        format = GL_RGB;
    else if (nrChannels == 4)
        format = GL_RGBA;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

    // Free resources
    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureId;
}

const char* getVertexShaderSource()
{
    return
        "#version 330 core\n"                          // GLSL version
        "layout (location = 0) in vec3 aPos;"          // Vertex position input
        "layout (location = 1) in vec3 aColor;"        // Vertex color input
        ""
        "uniform mat4 worldMatrix;"                    // Model transformation
        "uniform mat4 viewMatrix = mat4(1.0);"         // Default identity view
        "uniform mat4 projectionMatrix = mat4(1.0);"   // Default identity projection
        ""
        "out vec3 vertexColor;"                        // Output color to fragment shader
        "void main()"
        "{"
        "   vertexColor = aColor;"                     // Pass color to next stage
        "   mat4 modelViewProjection = projectionMatrix * viewMatrix * worldMatrix;"
        "   gl_Position = modelViewProjection * vec4(aPos.x, aPos.y, aPos.z, 1.0);" // Final screen position
        "}";
}

const char* getFragmentShaderSource()
{
    return
        "#version 330 core\n"                          // GLSL version
        "in vec3 vertexColor;"                         // Color received from vertex shader
        "out vec4 FragColor;"                          // Final pixel color
        "void main()"
        "{"
        "   FragColor = vec4(vertexColor.r, vertexColor.g, vertexColor.b, 1.0f);" // Output color with full opacity
        "}";
}

const char* getTexturedVertexShaderSource()
{
    return
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;"              // Vertex position
    "layout (location = 1) in vec3 aColor;"            // Vertex color
    "layout (location = 2) in vec2 aUV;"               // Vertex texture coordinates
    "layout (location = 3) in vec3 aNormal;"
    ""
    "uniform mat4 worldMatrix;"                        // Object transform
    "uniform mat4 viewMatrix;"                         // Camera transform
    "uniform mat4 projectionMatrix;"                   // Perspective projection
    "uniform mat4 lightVP;"
    ""
    "out vec3 vertexColor;"                            // Pass-through color
    "out vec2 vertexUV;"                               // Pass-through UV
    "out vec3 fragPos;"                                // Position in world space (for lighting)
    "out vec3 worldNormal;"
    "out vec4 lightSpacePos;"
    ""
    "void main()"
    "{"
    "   vertexColor = aColor;"                         // Forward color
    "   vertexUV = aUV;"                               // Forward UV
    "   "
    "   vec4 worldPos = worldMatrix * vec4(aPos, 1.0);" // Compute world space position
    "   fragPos = vec3(worldPos);"                     // Store for lighting calculations
    "   "
    "   mat3 normalMatrix = transpose(inverse(mat3(worldMatrix)));"
    "   worldNormal = normalize(normalMatrix * aNormal);"
    "   "
    "   lightSpacePos = lightVP * worldPos;"
    "   "
    "   gl_Position = projectionMatrix * viewMatrix * worldPos;" // Final screen-space position
    "}";
}

const char* getTexturedFragmentShaderSource()
{
    return
    "#version 330 core\n"
    "in vec3 worldNormal;\n"
    "in vec3 fragPos;\n"
    "in vec2 vertexUV;\n"
    "in vec4 lightSpacePos;\n"
    "\n"
    "uniform sampler2D textureSampler;\n"
    "uniform sampler2D shadowMap;\n"
    "uniform vec3 lightPos;\n"
    "uniform vec3 lightColor;\n"
    "uniform vec3 viewPos;\n"
    "uniform float specularStrength;\n"
    "uniform float shininess;\n"
    "// Spotlight params\n"
    "uniform vec3  spotDir;\n"
    "uniform float spotCutoff;\n"
    "uniform float spotOuterCutoff;\n"
    "// Shadow params\n"
    "uniform float shadowBias;\n"
    "uniform float shadowTexelSize;\n"
    "\n"
    "out vec4 FragColor;\n"
    "\n"
    "float calcShadowPCF(vec4 lsPos){\n"
    "    // perspective divide -> [0,1]\n"
    "    vec3 proj = lsPos.xyz / lsPos.w;\n"
    "    proj = proj * 0.5 + 0.5;\n"
    "    if (proj.z > 1.0) return 0.0; // outside light frustum\n"
    "    float shadow = 0.0;\n"
    "    // 3x3 PCF\n"
    "    for(int x=-1; x<=1; x++){\n"
    "      for(int y=-1; y<=1; y++){\n"
    "        float pcfDepth = texture(shadowMap, proj.xy + vec2(x,y)*shadowTexelSize).r;\n"
    "        // Make bias larger at shallow angles (reduces acne without over-biasing)\n"
    "        float ndotl     = max(dot(normalize(worldNormal), normalize(lightPos - fragPos)), 0.0);\n"
    "        float slopeBias = shadowBias * (1.0 - ndotl);   // 0..shadowBias\n"
    "        float current   = proj.z - (shadowBias + slopeBias);\n"
    "        shadow += current > pcfDepth ? 1.0 : 0.0;\n"
    "      }\n"
    "    }\n"
    "    return shadow / 9.0;\n"
    "}\n"
    "\n"
    "void main()\n"
    "{\n"
    "   vec3 norm = normalize(worldNormal);\n"
    "   vec3 L = normalize(lightPos - fragPos);\n"
    "   vec3 V = normalize(viewPos  - fragPos);\n"
    "   vec3 R = reflect(-L, norm);\n"
    "\n"
    "   // Distance attenuation\n"
    "   float dist = length(lightPos - fragPos);\n"
    "   float attenuation = 1.0 / (1.0 + 0.22*dist + 0.20*dist*dist);\n"
    "\n"
    "   // Spotlight falloff\n"
    "   float theta = dot(L, normalize(-spotDir));\n"
    "   float epsilon = spotCutoff - spotOuterCutoff;\n"
    "   float spot = clamp((theta - spotOuterCutoff) / max(epsilon, 1e-4), 0.0, 1.0);\n"
    "\n"
    "   float diff = max(dot(norm, L), 0.0);\n"
    "   float spec = pow(max(dot(V, R), 0.0), shininess);\n"
    "   vec3 ambient  = 0.05 * lightColor;\n"
    "   vec3 diffuse  = diff * lightColor;\n"
    "   vec3 specular = specularStrength * spec * lightColor;\n"
    "   vec3 lit = (ambient + (diffuse + specular) * spot) * attenuation;\n"
    "\n"
    "   // Shadows (darken diffuse+spec, keep ambient)\n"
    "   float shadow = calcShadowPCF(lightSpacePos);\n"
    "   vec3 litShadowed = ambient + (diffuse + specular) * (1.0 - shadow) * attenuation * spot;\n"
    "\n"
    "   vec3 tex = texture(textureSampler, vertexUV).rgb;\n"
    "   FragColor = vec4(litShadowed * tex, 1.0);\n"
    "}\n";
}

const char* getDepthVertexShaderSource()
{
    return
    "#version 330 core\n"
    "layout(location=0) in vec3 aPos;\n"
    "uniform mat4 worldMatrix;\n"
    "uniform mat4 lightVP;\n"
    "void main(){\n"
    "  gl_Position = lightVP * worldMatrix * vec4(aPos,1.0);\n"
    "}\n";
}

const char* getDepthFragmentShaderSource()
{
    return
    "#version 330 core\n"
    "void main(){}\n"; // depth-only
}

int compileAndLinkShaders(const char* vertexShaderSource, const char* fragmentShaderSource)
{
    // compile and link shader program
    // return shader program id
    // ------------------------------------

    // vertex shader
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // fragment shader
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // link shaders
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

GLuint setupModelVBO_OBJ(const std::string& path, int& vertexCount) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

    // Loader: fills vertices/normals/uvs as non-indexed arrays
    if (!loadOBJ(path.c_str(), vertices, normals, uvs)) {
        std::cerr << "Failed to load OBJ: " << path << std::endl;
        vertexCount = 0;
        return 0;
    }

    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Positions -> location 0
    GLuint vboPos;
    glGenBuffers(1, &vboPos);
    glBindBuffer(GL_ARRAY_BUFFER, vboPos);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    // UVs -> location 2 (your shader expects aUV at 2)
    if (!uvs.empty()) {
        GLuint vboUV;
        glGenBuffers(1, &vboUV);
        glBindBuffer(GL_ARRAY_BUFFER, vboUV);
        glBufferData(GL_ARRAY_BUFFER, uvs.size()*sizeof(glm::vec2), uvs.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(2);
    } else {
        // no UVs: disable attrib 2
        glDisableVertexAttribArray(2);
    }

    // Normals -> location 3 (your shader expects aNormal at 3)
    if (!normals.empty()) {
        GLuint vboNrm;
        glGenBuffers(1, &vboNrm);
        glBindBuffer(GL_ARRAY_BUFFER, vboNrm);
        glBufferData(GL_ARRAY_BUFFER, normals.size()*sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(3);
    } else {
        // no normals: disable attrib 3 (lighting will be wrong)
        glDisableVertexAttribArray(3);
    }

    // Provide a constant color for location 1 (white), since we don’t have per-vertex colors from OBJ
    glDisableVertexAttribArray(1);
    glVertexAttrib3f(1, 1.0f, 1.0f, 1.0f);

    glBindVertexArray(0);

    vertexCount = static_cast<int>(vertices.size());
    return VAO;
}

// Vertex definition for a textured + colored cube
struct TexturedColoredVertex
{
    TexturedColoredVertex(vec3 _position, vec3 _color, vec2 _uv, vec3 _normal)
    : position(_position), color(_color), uv(_uv), normal(_normal) {}
    
    vec3 position;
    vec3 color;
    vec2 uv;
    vec3 normal;
};

// Textured Cube model
const TexturedColoredVertex texturedCubeVertexArray[] = {  // position,                            color
    TexturedColoredVertex(vec3(-0.5f,-0.5f,-0.5f), vec3(1.0f, 0.0f, 0.0f), vec2(0.0f, 0.0f), vec3(-1,0,0)), //left - red
    TexturedColoredVertex(vec3(-0.5f,-0.5f, 0.5f), vec3(1.0f, 0.0f, 0.0f), vec2(0.0f, 1.0f), vec3(-1,0,0)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f, 0.5f), vec3(1.0f, 0.0f, 0.0f), vec2(1.0f, 1.0f), vec3(-1,0,0)),
    
    TexturedColoredVertex(vec3(-0.5f,-0.5f,-0.5f), vec3(1.0f, 0.0f, 0.0f), vec2(0.0f, 0.0f), vec3(-1,0,0)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f, 0.5f), vec3(1.0f, 0.0f, 0.0f), vec2(1.0f, 1.0f), vec3(-1,0,0)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f,-0.5f), vec3(1.0f, 0.0f, 0.0f), vec2(1.0f, 0.0f), vec3(-1,0,0)),
    
    TexturedColoredVertex(vec3( 0.5f, 0.5f,-0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f), vec3(0,0,-1)), // far - blue
    TexturedColoredVertex(vec3(-0.5f,-0.5f,-0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 0.0f), vec3(0,0,-1)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f,-0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 1.0f), vec3(0,0,-1)),
    
    TexturedColoredVertex(vec3( 0.5f, 0.5f,-0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f), vec3(0,0,-1)),
    TexturedColoredVertex(vec3( 0.5f,-0.5f,-0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 0.0f), vec3(0,0,-1)),
    TexturedColoredVertex(vec3(-0.5f,-0.5f,-0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 0.0f), vec3(0,0,-1)),
    
    TexturedColoredVertex(vec3( 0.5f,-0.5f, 0.5f), vec3(0.0f, 1.0f, 1.0f), vec2(1.0f, 1.0f), vec3(0,-1,0)), // bottom - turquoise
    TexturedColoredVertex(vec3(-0.5f,-0.5f,-0.5f), vec3(0.0f, 1.0f, 1.0f), vec2(0.0f, 0.0f), vec3(0,-1,0)),
    TexturedColoredVertex(vec3( 0.5f,-0.5f,-0.5f), vec3(0.0f, 1.0f, 1.0f), vec2(1.0f, 0.0f), vec3(0,-1,0)),
    
    TexturedColoredVertex(vec3( 0.5f,-0.5f, 0.5f), vec3(0.0f, 1.0f, 1.0f), vec2(1.0f, 1.0f), vec3(0,-1,0)),
    TexturedColoredVertex(vec3(-0.5f,-0.5f, 0.5f), vec3(0.0f, 1.0f, 1.0f), vec2(0.0f, 1.0f), vec3(0,-1,0)),
    TexturedColoredVertex(vec3(-0.5f,-0.5f,-0.5f), vec3(0.0f, 1.0f, 1.0f), vec2(0.0f, 0.0f), vec3(0,-1,0)),
    
    TexturedColoredVertex(vec3(-0.5f, 0.5f, 0.5f), vec3(0.0f, 1.0f, 0.0f), vec2(0.0f, 1.0f), vec3(0,0,1)), // near - green
    TexturedColoredVertex(vec3(-0.5f,-0.5f, 0.5f), vec3(0.0f, 1.0f, 0.0f), vec2(0.0f, 0.0f), vec3(0,0,1)),
    TexturedColoredVertex(vec3( 0.5f,-0.5f, 0.5f), vec3(0.0f, 1.0f, 0.0f), vec2(1.0f, 0.0f), vec3(0,0,1)),
    
    TexturedColoredVertex(vec3( 0.5f, 0.5f, 0.5f), vec3(0.0f, 1.0f, 0.0f), vec2(1.0f, 1.0f), vec3(0,0,1)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f, 0.5f), vec3(0.0f, 1.0f, 0.0f), vec2(0.0f, 1.0f), vec3(0,0,1)),
    TexturedColoredVertex(vec3( 0.5f,-0.5f, 0.5f), vec3(0.0f, 1.0f, 0.0f), vec2(1.0f, 0.0f), vec3(0,0,1)),
    
    TexturedColoredVertex(vec3( 0.5f, 0.5f, 0.5f), vec3(1.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f), vec3(1,0,0)), // right - purple
    TexturedColoredVertex(vec3( 0.5f,-0.5f,-0.5f), vec3(1.0f, 0.0f, 1.0f), vec2(0.0f, 0.0f), vec3(1,0,0)),
    TexturedColoredVertex(vec3( 0.5f, 0.5f,-0.5f), vec3(1.0f, 0.0f, 1.0f), vec2(1.0f, 0.0f), vec3(1,0,0)),
    
    TexturedColoredVertex(vec3( 0.5f,-0.5f,-0.5f), vec3(1.0f, 0.0f, 1.0f), vec2(0.0f, 0.0f), vec3(1,0,0)),
    TexturedColoredVertex(vec3( 0.5f, 0.5f, 0.5f), vec3(1.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f), vec3(1,0,0)),
    TexturedColoredVertex(vec3( 0.5f,-0.5f, 0.5f), vec3(1.0f, 0.0f, 1.0f), vec2(0.0f, 1.0f), vec3(1,0,0)),
    
    TexturedColoredVertex(vec3( 0.5f, 0.5f, 0.5f), vec3(1.0f, 1.0f, 0.0f), vec2(1.0f, 1.0f), vec3(0,1,0)), // top - yellow
    TexturedColoredVertex(vec3( 0.5f, 0.5f,-0.5f), vec3(1.0f, 1.0f, 0.0f), vec2(1.0f, 0.0f), vec3(0,1,0)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f,-0.5f), vec3(1.0f, 1.0f, 0.0f), vec2(0.0f, 0.0f), vec3(0,1,0)),
    
    TexturedColoredVertex(vec3( 0.5f, 0.5f, 0.5f), vec3(1.0f, 1.0f, 0.0f), vec2(1.0f, 1.0f), vec3(0,1,0)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f,-0.5f), vec3(1.0f, 1.0f, 0.0f), vec2(0.0f, 0.0f), vec3(0,1,0)),
    TexturedColoredVertex(vec3(-0.5f, 0.5f, 0.5f), vec3(1.0f, 1.0f, 0.0f), vec2(0.0f, 1.0f), vec3(0,1,0))
};

int createTexturedCubeVertexArrayObject()
{
    // Create a vertex array
    GLuint vertexArrayObject;
    glGenVertexArrays(1, &vertexArrayObject);
    glBindVertexArray(vertexArrayObject);
    
    // Upload Vertex Buffer to the GPU, keep a reference to it (vertexBufferObject)
    GLuint vertexBufferObject;
    glGenBuffers(1, &vertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texturedCubeVertexArray), texturedCubeVertexArray, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0,                   // attribute 0 matches aPos in Vertex Shader
                          3,                   // size
                          GL_FLOAT,            // type
                          GL_FALSE,            // normalized?
                          sizeof(TexturedColoredVertex), // stride - each vertex contain 2 vec3 (position, color)
                          (void*)0             // array buffer offset
                          );
    glEnableVertexAttribArray(0);
    
    
    glVertexAttribPointer(1,                            // attribute 1 matches aColor in Vertex Shader
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(TexturedColoredVertex),
                          (void*)sizeof(vec3)      // color is offseted a vec3 (comes after position)
                          );
    glEnableVertexAttribArray(1);
    
    glVertexAttribPointer(2,                            // attribute 2 matches aUV in Vertex Shader
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(TexturedColoredVertex),
                          (void*)(2*sizeof(vec3))      // uv is offseted by 2 vec3 (comes after position and color)
                          );
    glEnableVertexAttribArray(2);
    
    glVertexAttribPointer(3, 
                          3, 
                          GL_FLOAT, 
                          GL_FALSE, 
                          sizeof(TexturedColoredVertex), 
                          (void*)(2*sizeof(vec3) + sizeof(vec2))
                          );
    glEnableVertexAttribArray(3);

    return vertexArrayObject;
}

struct Door {
    glm::vec3 basePos;          // hinge position in world (x,y,z)
    float angleDeg = 0.0f;      // current angle
    float targetDeg = 0.0f;     // desired angle (0 = closed, 90 = open)
    float speedDeg = 120.0f;    // deg/sec
    float baseYawDeg = 0.0f;    // static yaw of the door frame

    // Knob twist animation
    float knobAngleDeg = 0.0f;
    float knobTargetDeg = 0.0f;  // e.g., 25° when opening, 0° when released
    float knobSpeedDeg  = 360.0f; // fast twist
};

Door gDoorFinish;                   // door at the finish
bool  gFinishDoorUnlocked = false;  // unlock when you reach some trigger

// Uploads projection matrix to the shader
void setProjectionMatrix(int shaderProgram, mat4 projectionMatrix)
{
    glUseProgram(shaderProgram);
    GLuint projectionMatrixLocation = glGetUniformLocation(shaderProgram, "projectionMatrix");
    glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]);
}

// Uploads view matrix to the shader
void setViewMatrix(int shaderProgram, mat4 viewMatrix)
{
    glUseProgram(shaderProgram);
    GLuint viewMatrixLocation = glGetUniformLocation(shaderProgram, "viewMatrix");
    glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]);
}

// Uploads world (model) matrix to the shader
void setWorldMatrix(int shaderProgram, mat4 worldMatrix)
{
    glUseProgram(shaderProgram);
    GLuint worldMatrixLocation = glGetUniformLocation(shaderProgram, "worldMatrix");
    glUniformMatrix4fv(worldMatrixLocation, 1, GL_FALSE, &worldMatrix[0][0]);
}

const int MAZE_WIDTH = 20;
const int MAZE_HEIGHT = 20;

// A 20x20 maze represented as an array of strings
// '#' = wall, 'S' = start, 'F' = finish, ' ' = path
const char* maze[MAZE_HEIGHT] = {
    "####################",
    "##  #####  #####  ##",
    "#                  #",
    "#                  F",
    "##  #####  #####  ##",
    "##  #          #  ##",
    "##  # ###  ### #  ##",
    "##  # #      # #  ##",
    "##  # #      # #  ##",
    "#                  #",
    "#                 S#",
    "##  # #      # #  ##",
    "##  # #      # #  ##",
    "##  # ###  ### #  ##",
    "##  #          #  ##",
    "##  #####  #####  ##",
    "#                  #",
    "#                  #",
    "##  #####  #####  ##",
    "####################"
};

glm::vec3 startPosition;    // Player starting point in the maze
glm::vec3 finishPosition;   // Maze exit goal position

// Flame light settings (static at entrance)
vec3 flameLightColor(1.0f, 0.5f, 0.2f); // orange flame color
vec3 flameBasePosition;                 // will be initialized once we find 'S' in the maze

// Shadow mapping
GLuint gShadowFBO = 0;
GLuint gShadowDepthTex = 0;
const unsigned int SHADOW_RES = 1024; // 1024 (use 2048 if you have perf headroom)

// Depth-only shader (light pass)
int depthShaderProgram = 0;

float gTorchIntensity = 3.0f; // default starting intensity

// Simple grid collision: treat out-of-bounds as a wall; '#' blocks movement
inline bool CollidesWithWall(const glm::vec3& p) {
    int mx = static_cast<int>(p.x);
    int mz = static_cast<int>(p.z);
    if (mx < 0 || mx >= MAZE_WIDTH || mz < 0 || mz >= MAZE_HEIGHT) return true;
    return maze[mz][mx] == '#';
}

int main(int argc, char*argv[])
{
    // Initialize GLFW and OpenGL version
    glfwInit();
    
    #if defined(PLATFORM_OSX)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #else
        // On windows, we set OpenGL version to 2.1, to support more hardware
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    #endif

    // Create Window and rendering context using GLFW, resolution is 800x600
    GLFWwindow* window = glfwCreateWindow(800, 600, "Comp371 - Maze Game", NULL, NULL);
    if (window == NULL)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // First‑person mouse: hide & lock cursor to the window
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to create GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Load textures used in the scene
    GLuint groundTextureID = loadTexture("Textures/ground_0044_color_1k.jpg");
    GLuint wallTextureID = loadTexture("Textures/ground_0014_subsurface_1k.jpg");
    GLuint statueTextureID = loadTexture("Textures/rock_0005_color_1k.jpg");
    GLuint doorTextureID = loadTexture("Textures/wood_0032_color_1k.jpg");
    GLuint knobTextureID = loadTexture("Textures/rock_0005_color_1k.jpg");
    
    // Black background
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Build shader programs (basic color + textured/lighting)
    int colorShaderProgram = compileAndLinkShaders(getVertexShaderSource(), getFragmentShaderSource());
    int texturedShaderProgram = compileAndLinkShaders(getTexturedVertexShaderSource(), getTexturedFragmentShaderSource());

    glUseProgram(texturedShaderProgram);    // set as current before setting its uniforms

    // Depth‑only shader for shadow pass
    depthShaderProgram = compileAndLinkShaders(getDepthVertexShaderSource(), getDepthFragmentShaderSource());

    // ---- Shadow map setup (depth texture + FBO) ----
    glGenTextures(1, &gShadowDepthTex);
    glBindTexture(GL_TEXTURE_2D, gShadowDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_RES, SHADOW_RES, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // precise lookups for PCF
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border[4] = {1.0f, 1.0f, 1.0f, 1.0f}; // treat outside as lit
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

    glGenFramebuffers(1, &gShadowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, gShadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gShadowDepthTex, 0);
    glDrawBuffer(GL_NONE);  // depth‑only
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Shadow FBO incomplete\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Static scene‑wide uniforms for the textured shader
    glUseProgram(texturedShaderProgram);
    glUniform1f(glGetUniformLocation(texturedShaderProgram, "shadowBias"), 0.0015f);
    glUniform1f(glGetUniformLocation(texturedShaderProgram, "shadowTexelSize"), 1.0f / float(SHADOW_RES));

    // Cache locations for per‑frame lighting updates
    GLint lightPosLocation = glGetUniformLocation(texturedShaderProgram, "lightPos");
    GLint lightColorLocation = glGetUniformLocation(texturedShaderProgram, "lightColor");
    GLint viewPosLocation = glGetUniformLocation(texturedShaderProgram, "viewPos");

    // Specular material params (Phong)
    GLint specularStrengthLocation = glGetUniformLocation(texturedShaderProgram, "specularStrength");
    GLint shininessLocation        = glGetUniformLocation(texturedShaderProgram, "shininess");

    // Reasonable defaults for shiny surfaces
    glUniform1f(specularStrengthLocation, 0.5f); // 0.2–1.0 typical
    glUniform1f(shininessLocation, 32.0f);       // higher = tighter highlight
    
    // --- Camera bootstrap (position, orientation, speeds) ---
    vec3 cameraPosition(1.0f, 1.0f, 1.0f);
    vec3 cameraLookAt(0.0f, 0.0f, -1.0f);
    vec3 cameraUp(0.0f, 1.0f, 0.0f);
    
    // Camera settings
    float cameraSpeed = 1.5f;                   // walking
    float cameraFastSpeed = 2 * cameraSpeed;    // sprint
    float cameraHorizontalAngle = 90.0f;        // yaw
    float cameraVerticalAngle = 0.0f;           // pitch
    
    // Set projection matrix for shader
    mat4 projectionMatrix = glm::perspective(70.0f,            // field of view in degrees
                                             800.0f / 600.0f,  // aspect ratio
                                             0.01f, 100.0f);   // near and far (near > 0)
    
    // Set initial view matrix
    mat4 viewMatrix = lookAt(cameraPosition,                    // eye
                             cameraPosition + cameraLookAt,     // center
                             cameraUp );                        // up
    
    // Set View and Projection matrices on both shaders
    setViewMatrix(colorShaderProgram, viewMatrix);
    setViewMatrix(texturedShaderProgram, viewMatrix);
    setProjectionMatrix(colorShaderProgram, projectionMatrix);
    setProjectionMatrix(texturedShaderProgram, projectionMatrix);

    // Upload geometry to the GPU
    int texturedCubeVAO = createTexturedCubeVertexArrayObject();

    // --- Statue (OBJ) ---
    int statueVertexCount = 0;
    GLuint statueVAO = setupModelVBO_OBJ("Models/heracles.obj", statueVertexCount);
    if (!statueVAO || statueVertexCount == 0) {
        std::cerr << "Failed to load statue OBJ\n";
    }

    // Place & orient it at maze center
    float statueScale = 0.08f; // tweak
    mat4 statueWorld =
        glm::translate(mat4(1.0f), vec3(MAZE_WIDTH/2.0f, 0.5f, MAZE_HEIGHT/2.0f)) *
        glm::rotate(mat4(1.0f), glm::radians(-90.0f), vec3(1,0,0)) *    // fix lay-down orientation
        glm::scale(mat4(1.0f), vec3(statueScale));

    // Collision proxy for the statue (XZ cylinder)
    glm::vec2 gStatueCenter = glm::vec2(MAZE_WIDTH / 2.0f, MAZE_HEIGHT / 2.0f);
    float gStatueRadius = 0.6f; // depends on mesh scale

    // Timing + mouse input bootstrap
    float lastFrameTime = glfwGetTime();    // For delta time
    int lastMouseLeftState = GLFW_RELEASE;
    double lastMousePosX, lastMousePosY;
    glfwGetCursorPos(window, &lastMousePosX, &lastMousePosY);
    
    // Enable OpenGL States
    glEnable(GL_CULL_FACE);     // Back-face culling
    glEnable(GL_DEPTH_TEST);    // Depth testing for proper 3D rendering

    // Use Textured Cube VAO
    glBindVertexArray(texturedCubeVAO);

    // Parse maze once to locate Start/Finish and place camera/light
    for (int row = 0; row < MAZE_HEIGHT; ++row) {
        for (int col = 0; col < MAZE_WIDTH; ++col) {
            if (maze[row][col] == 'S') {
                startPosition     = glm::vec3(col + 0.5f, 0.0f, row + 0.5f);
                flameBasePosition = startPosition + vec3(0.0f, 0.5f, 0.0f);
                // place the camera a bit above the floor at S
                cameraPosition    = startPosition; 
                cameraPosition.y  = 1.0f;
            }
            else if (maze[row][col] == 'F') {
                finishPosition = glm::vec3(col + 0.5f, 0.0f, row + 0.5f);
            }
        }
    }

    // Finish door: hinge on the left edge of the 'F' cell, yaw aligns door frame
    gDoorFinish.basePos = finishPosition + glm::vec3(-0.5f, 0.0f, 0.5f);
    gDoorFinish.baseYawDeg = 90.0f;
    gDoorFinish.angleDeg = 0.0f;    // closed
    gDoorFinish.targetDeg = 0.0f;   // closed target

    // Entering Main Game Loop
    while(!glfwWindowShouldClose(window))
    {
        // Frame time calculation
        float dt = glfwGetTime() - lastFrameTime;
        lastFrameTime += dt;
        glfwPollEvents();

        // Smoothly animate door rotation towards target
        auto updateDoorHier = [&](Door& d, float dt){
            // 1) knob first
            float kdiff = d.knobTargetDeg - d.knobAngleDeg;
            float kstep = glm::clamp(kdiff, -d.knobSpeedDeg * dt, d.knobSpeedDeg * dt);
            d.knobAngleDeg += kstep;

            // 2) when knob is mostly twisted, rotate the slab
            float kprogress = (fabsf(d.knobTargetDeg) < 1e-3f) ? 1.0f
                            : glm::clamp(fabsf(d.knobAngleDeg) / fabsf(d.knobTargetDeg), 0.0f, 1.0f);
            if (kprogress > 0.9f) {
                float ddiff = d.targetDeg - d.angleDeg;
                float dstep = glm::clamp(ddiff, -d.speedDeg * dt, d.speedDeg * dt);
                d.angleDeg += dstep;
            }

            // optional: when door reaches its target, relax knob back to 0
            if (fabsf(d.targetDeg - d.angleDeg) < 0.5f) {
                d.knobTargetDeg = 0.0f;
            }
        };
        updateDoorHier(gDoorFinish, dt);

        // Mouse Input: update camera angles
        double mousePosX, mousePosY;
        glfwGetCursorPos(window, &mousePosX, &mousePosY);
        double dx = mousePosX - lastMousePosX;
        double dy = mousePosY - lastMousePosY;
        lastMousePosX = mousePosX;
        lastMousePosY = mousePosY;

        // Convert mouse movement to spherical angles
        const float cameraAngularSpeed = 20.0f;
        cameraHorizontalAngle -= dx * cameraAngularSpeed * dt;
        cameraVerticalAngle   -= dy * cameraAngularSpeed * dt;
        
        // Clamp vertical camera angle to [-85, 85] degrees
        cameraVerticalAngle = std::max(-85.0f, std::min(85.0f, cameraVerticalAngle));
        
        // Convert angles to direction vectors
        float theta = radians(cameraHorizontalAngle);
        float phi = radians(cameraVerticalAngle);
        cameraLookAt = vec3(cosf(phi)*cosf(theta), sinf(phi), -cosf(phi)*sinf(theta));
        vec3 cameraSideVector = glm::cross(cameraLookAt, vec3(0.0f, 1.0f, 0.0f));
        cameraSideVector = glm::normalize(cameraSideVector);

        // --- Keyboard movement ---
        bool fastCam = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        float currentCameraSpeed = (fastCam) ? cameraFastSpeed : cameraSpeed;
        vec3 proposedPosition = cameraPosition;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            proposedPosition += cameraLookAt * dt * currentCameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            proposedPosition -= cameraLookAt * dt * currentCameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            proposedPosition += cameraSideVector * dt * currentCameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            proposedPosition -= cameraSideVector * dt * currentCameraSpeed;

        // --- Interaction: open/close door ---
        auto nearDoor = [&](const Door& d){
            return glm::length(glm::vec2(cameraPosition.x - d.basePos.x,
                                        cameraPosition.z - d.basePos.z)) < 1.8f;
        };
        // Toggle start door with E
        static bool eWasDown = false;
        bool eDown = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
        if (eDown && !eWasDown) {                 // on key press (not hold)
            if (nearDoor(gDoorFinish)) {          // require proximity to interact
                // toggle target angle
                bool opening = (gDoorFinish.targetDeg >= -1.0f); // currently aiming to closed?
                gDoorFinish.targetDeg   = opening ? -90.0f : 0.0f;  // swing
                gDoorFinish.knobTargetDeg = opening ? 25.0f : 15.0f; // a nice twist pulse
            }
        }
        eWasDown = eDown;

        // --- Torch brightness controls ---
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) 
            gTorchIntensity = std::min(gTorchIntensity + 0.05f, 6.0f);
        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET)  == GLFW_PRESS) 
            gTorchIntensity = std::max(gTorchIntensity - 0.05f, 0.5f);

        // Keep player at constant eye height
        proposedPosition.y = 1.0f;

        // --- Collision: walls ---
        if (!CollidesWithWall(proposedPosition)) {
            // --- Collision: statue ---
            glm::vec2 camXZ(proposedPosition.x, proposedPosition.z);
            glm::vec2 delta = camXZ - gStatueCenter;
            float d = glm::length(delta);

            if (d < gStatueRadius) {
                glm::vec2 n = (d > 1e-4f) ? (delta / d) : glm::vec2(1.0f, 0.0f);
                glm::vec2 projectedXZ = gStatueCenter + n * gStatueRadius;
                glm::vec3 pushedPos = proposedPosition;
                pushedPos.x = projectedXZ.x;
                pushedPos.z = projectedXZ.y;

                // Only accept the push if it doesn't put us into a wall
                if (!CollidesWithWall(pushedPos)) {
                    proposedPosition = pushedPos;
                } else {
                    // If push would hit a wall, stay at current cameraPosition (cancel move)
                    proposedPosition = cameraPosition;
                }
            }

            // After statue resolution, double-check walls (in case we clipped into one)
            if (!CollidesWithWall(proposedPosition)) {
                cameraPosition = proposedPosition;
            } else {
                cameraPosition.y = 1.0f; // keep height even if we don't move
            }

        } else {
            // Hit a wall – don't move, just keep height locked
            cameraPosition.y = 1.0f;
        }

        // --- Collision: door slab ---
        auto blockByDoor = [&](const Door& d, glm::vec3& pos){
            if (d.angleDeg > 80.0f) return; // open enough to pass
            float w = 1.0f;
            float yaw = glm::radians(d.baseYawDeg + d.angleDeg);

            glm::vec3 A = d.basePos; // hinge
            glm::vec3 B = d.basePos + glm::vec3(
                glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0,1,0)) * glm::vec4(w, 0, 0, 1)
            );

            glm::vec2 p(pos.x, pos.z), a(A.x, A.z), b(B.x, B.z);
            glm::vec2 ab = b - a; float ab2 = glm::dot(ab, ab);
            float tproj = (ab2 > 1e-5f) ? glm::clamp(glm::dot(p - a, ab) / ab2, 0.0f, 1.0f) : 0.0f;
            glm::vec2 closest = a + tproj * ab;

            float radius = 0.20f;
            glm::vec2 diff = p - closest;
            float d2 = glm::dot(diff, diff);
            if (d2 < radius * radius) {
                float dlen = sqrtf(d2);
                glm::vec2 n = (dlen > 1e-4f) ? (diff / dlen) : glm::vec2(1,0);
                glm::vec2 newP = closest + n * radius;
                glm::vec3 pushed = pos; pushed.x = newP.x; pushed.z = newP.y;
                if (!CollidesWithWall(pushed)) pos = pushed;
            }
        };
        blockByDoor(gDoorFinish, proposedPosition);
        if (!CollidesWithWall(proposedPosition)) cameraPosition = proposedPosition;

        // ESC to quit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // --- Update view matrices ---
        mat4 viewMatrix(1.0f);
        viewMatrix = lookAt(cameraPosition, cameraPosition + cameraLookAt, cameraUp );
        setViewMatrix(colorShaderProgram, viewMatrix);
        setViewMatrix(texturedShaderProgram, viewMatrix);

        // --- Torch as a spotlight + shadow caster ---
        vec3 dynamicLightColor = vec3(1.0f, 0.95f, 0.85f) * gTorchIntensity;
        vec3 torchPos = cameraPosition + cameraLookAt * 0.4f + vec3(0.0f, 0.3f, 0.0f);
        vec3 torchDir = normalize(cameraLookAt);

        // Build light matrices for the shadow pass and scene pass
        float lightNearPlane = 0.05f;
        float lightFarPlane  = 20.0f;
        mat4 lightProjMatrix = glm::perspective(glm::radians(30.0f), 1.0f, lightNearPlane, lightFarPlane);
        mat4 lightViewMatrix = glm::lookAt(torchPos, torchPos + torchDir, vec3(0,1,0));
        mat4 lightVP         = lightProjMatrix * lightViewMatrix;

        glUseProgram(texturedShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(texturedShaderProgram, "lightVP"), 1, GL_FALSE, &lightVP[0][0]);
        glUniform3fv(lightColorLocation, 1, &dynamicLightColor[0]);
        glUniform3fv(lightPosLocation,   1, &torchPos[0]);
        glUniform3fv(viewPosLocation,    1, &cameraPosition[0]);
        glUniform3fv(glGetUniformLocation(texturedShaderProgram, "spotDir"), 1, &torchDir[0]);
        glUniform1f(glGetUniformLocation(texturedShaderProgram, "spotCutoff"), cos(radians(16.0f)));
        glUniform1f(glGetUniformLocation(texturedShaderProgram, "spotOuterCutoff"), cos(radians(22.0f)));

        // Helper for placing door in world space
        // Parent: hinge/frame transform (no scaling to keep child math clean)
        auto makeDoorParentMatrix = [&](const Door& d){
            glm::mat4 M(1.0f);
            // Move to hinge point at floor, lift to mid-height so child scales around center
            // For drawing children we’ll handle their local offsets.
            M = glm::translate(M, d.basePos);
            M = glm::rotate(M, glm::radians(d.baseYawDeg), glm::vec3(0,1,0));
            return M;
        };

        // Child 1: slab inherits parent, then rotates about hinge, then offsets to slab center, then scales
        auto makeDoorSlabMatrix = [&](const Door& d, float thickness){
            float w = 1.0f;   // hinge→far edge
            float h = 5.0f;   // slab height
            float t = thickness;

            glm::mat4 M = makeDoorParentMatrix(d);
            // rotate the slab around the hinge (local +Y)
            M = glm::rotate(M, glm::radians(d.angleDeg), glm::vec3(0,1,0));
            // move to slab center (half-width along local +X) and lift to mid-height
            M = glm::translate(M, glm::vec3(w * 0.5f, h * 0.5f, 0.0f));
            // scale cube -> slab
            M = glm::scale(M, glm::vec3(w, h, t));
            return M;
        };

        // Child 2: knob inherits slab’s local space; we apply a small local twist for animation
        auto makeDoorKnobMatrix = [&](const Door& d){
            // Build from parent with same rotation as the slab (so it rides along),
            // then place a small cube where a knob would be, and twist it around its own axis.
            float h = 5.0f;
            float knobHeight = 1.0f;    // ~1m above floor
            float slabW = 1.0f;
            float outZ = -0.09f;         // offset outward from the door face

            glm::mat4 M = makeDoorParentMatrix(d);
            M = glm::rotate(M, glm::radians(d.angleDeg), glm::vec3(0,1,0));   // follow slab
            // Move to knob location: along the slab width (near far edge), and up from floor
            M = glm::translate(M, glm::vec3(slabW * 0.8f, knobHeight, outZ));
            // Local twist around X to simulate turning (or use Z/Y if you prefer)
            M = glm::rotate(M, glm::radians(d.knobAngleDeg), glm::vec3(0,0,1));
            // Small cube
            M = glm::scale(M, glm::vec3(0.08f, 0.08f, 0.08f));
            return M;
        };

        // === PASS 1: shadow map ===
        glUseProgram(depthShaderProgram);
        glViewport(0, 0, SHADOW_RES, SHADOW_RES);
        glBindFramebuffer(GL_FRAMEBUFFER, gShadowFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        // Send lightVP to depth shader
        glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "lightVP"), 1, GL_FALSE, &lightVP[0][0]);

        // (optional acne reduction during depth pass)
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);

        // Draw ground to depth
        {
            mat4 groundWorldMatrix = glm::translate(mat4(1.0f), vec3(MAZE_WIDTH / 2.0f, -0.01f, MAZE_HEIGHT / 2.0f)) *
                                    glm::scale(mat4(1.0f), vec3(MAZE_WIDTH, 0.02f, MAZE_HEIGHT));
            setWorldMatrix(depthShaderProgram, groundWorldMatrix);
            glBindVertexArray(texturedCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        // Draw walls to depth
        for (int row = 0; row < MAZE_HEIGHT; ++row) {
            for (int col = 0; col < MAZE_WIDTH; ++col) {
                if (maze[row][col] == '#') {
                    float wallHeight = 4.0f;
                    mat4 wallModel = glm::translate(mat4(1.0f), vec3(col + 0.5f, wallHeight/2.0f, row + 0.5f)) *
                                    glm::scale(mat4(1.0f), vec3(1.0f, wallHeight, 1.0f));
                    setWorldMatrix(depthShaderProgram, wallModel);
                    glDrawArrays(GL_TRIANGLES, 0, 36);
                }
            }
        }

        // Statue to depth (casts shadow)
        if (statueVAO && statueVertexCount > 0) {
            glBindVertexArray(statueVAO);
            setWorldMatrix(depthShaderProgram, statueWorld);
            glDrawArrays(GL_TRIANGLES, 0, statueVertexCount);
            glBindVertexArray(texturedCubeVAO); // <-- rebind cube VAO for anything else
        }

        // Finish door to depth
        {
            glm::mat4 doorSlabM = makeDoorSlabMatrix(gDoorFinish, 0.15f);
            setWorldMatrix(depthShaderProgram, doorSlabM);
            glBindVertexArray(texturedCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glm::mat4 knobM = makeDoorKnobMatrix(gDoorFinish);
            setWorldMatrix(depthShaderProgram, knobM);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        // Restore state & default framebuffer
        glCullFace(GL_BACK);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // === PASS 2: scene ===
        int fbw, fbh; glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(texturedShaderProgram);

        // (optional redundancy) ensure lightVP is set on scene shader before drawing
        glUniformMatrix4fv(glGetUniformLocation(texturedShaderProgram, "lightVP"), 1, GL_FALSE, &lightVP[0][0]);

        // Bind shadow map on unit 1
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, gShadowDepthTex);
        glUniform1i(glGetUniformLocation(texturedShaderProgram, "shadowMap"), 1);

        // Bind ground texture
        glActiveTexture(GL_TEXTURE0);
        GLuint textureLocation = glGetUniformLocation(texturedShaderProgram, "textureSampler");
        glBindTexture(GL_TEXTURE_2D, groundTextureID);
        glUniform1i(textureLocation, 0);                // Set our Texture sampler to user Texture Unit 0
        
        // Draw Maze ground
        mat4 groundWorldMatrix = glm::translate(mat4(1.0f), vec3(MAZE_WIDTH / 2.0f, -0.01f, MAZE_HEIGHT / 2.0f)) *
                                 glm::scale(mat4(1.0f), vec3(MAZE_WIDTH, 0.02f, MAZE_HEIGHT));
        setWorldMatrix(texturedShaderProgram, groundWorldMatrix);
        glDrawArrays(GL_TRIANGLES, 0, 36); // 36 vertices, starting at index 0
        
        // Draw Maze walls
        glBindTexture(GL_TEXTURE_2D, wallTextureID);
        for (int row = 0; row < MAZE_HEIGHT; ++row) {
            for (int col = 0; col < MAZE_WIDTH; ++col) {
                if (maze[row][col] == '#') {
                    // Draw wall with scaling
                    float wallHeight = 5.0f;
                    glm::mat4 wallModel = glm::translate(glm::mat4(1.0f), glm::vec3(col + 0.5f, wallHeight/2.0f, row + 0.5f)) *
                                        glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, wallHeight, 1.0f));
                    setWorldMatrix(texturedShaderProgram, wallModel);

                    glDrawArrays(GL_TRIANGLES, 0, 36);
                }
            }
        }

        // --- Statue in scene (receives & casts shadow) ---
        if (statueVAO && statueVertexCount > 0) {
            glUseProgram(texturedShaderProgram);

            // Bind statue texture on unit 0
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, statueTextureID);
            glUniform1i(glGetUniformLocation(texturedShaderProgram, "textureSampler"), 0);

            // World matrix for statue (you already computed statueWorld)
            setWorldMatrix(texturedShaderProgram, statueWorld);
            glBindVertexArray(statueVAO);
            glDrawArrays(GL_TRIANGLES, 0, statueVertexCount);

            // IMPORTANT: restore cube VAO & wall texture for anything drawn after
            glBindVertexArray(texturedCubeVAO);
            glBindTexture(GL_TEXTURE_2D, wallTextureID);
            glUniform1i(glGetUniformLocation(texturedShaderProgram, "textureSampler"), 0);
        }

        // ---- Door slab ----
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, doorTextureID);
        glUniform1i(glGetUniformLocation(texturedShaderProgram, "textureSampler"), 0);
        glm::mat4 doorSlabM = makeDoorSlabMatrix(gDoorFinish, 0.15f);
        setWorldMatrix(texturedShaderProgram, doorSlabM);
        glBindVertexArray(texturedCubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // ---- Door knob (different texture) ----
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, knobTextureID); // <- different look
        glUniform1i(glGetUniformLocation(texturedShaderProgram, "textureSampler"), 0);
        glm::mat4 knobM = makeDoorKnobMatrix(gDoorFinish); // use your face sign as needed
        setWorldMatrix(texturedShaderProgram, knobM);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // Draw colored geometry
        glUseProgram(colorShaderProgram);

        // Win Condition: Reaching Finish
        float distanceToFinish = glm::distance(glm::vec2(cameraPosition.x, cameraPosition.z), glm::vec2(finishPosition.x, finishPosition.z));
        if (distanceToFinish < 0.7f) {
            std::cout << "Congrats! You reached the end of the maze!" << std::endl;
            glfwSetWindowShouldClose(window, true); // Close the game window
        }

        // End Frame
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    
	return 0;
}