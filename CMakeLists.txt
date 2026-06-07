cmake_minimum_required(VERSION 3.10)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(OldVideoHDRemastererAI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# FetchContent for dependency management
include(FetchContent)

# cxxopts: Command line argument parsing
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.1.1
)
FetchContent_MakeAvailable(cxxopts)

# GLFW: Window management
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# OpenGL: Rendering
find_package(OpenGL REQUIRED)

# ImGui: GUI via FetchContent
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.1
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)
  add_library(imgui STATIC 
    ${imgui_SOURCE_DIR}/imgui.cpp 
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
endif()

# ncnn: High-performance neural network inference framework (FetchContent版を使用)
find_package(Vulkan REQUIRED)
option(NCNN_VULKAN "enable vulkan support" ON)

FetchContent_Declare(
  ncnn
  GIT_REPOSITORY https://github.com/Tencent/ncnn.git
  GIT_TAG        20240102
)
FetchContent_MakeAvailable(ncnn)

# OpenCV: Image and video processing (videostab は環境依存のため除外)
set(OpenCV_DIR "F:/MovieAPP/HDRemaster/opencv_dev/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio highgui)

include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/video_processor.cpp
    src/video_processor.h
    src/utils.cpp
    src/utils.h
    src/upscaler.cpp
    src/upscaler.h
    src/stabilizer.cpp
    src/stabilizer.h
    src/interpolator.cpp
    src/interpolator.h
    src/rife_warp.cpp
    src/rife_warp.h
    src/gui.cpp
    src/gui.h
    src/localization.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# NativeFileDialog-Extended
FetchContent_Declare(
  nfd
  GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
  GIT_TAG        v1.2.1
)
FetchContent_MakeAvailable(nfd)

# vcpkg の ncnn ヘッダーが FetchContent 版より優先されるのを防ぐため
# FetchContent 版の ncnn include を最優先にする
target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${ncnn_SOURCE_DIR}/src
    ${ncnn_BINARY_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cxxopts::cxxopts
        ncnn
        Vulkan::Vulkan
        ${OpenCV_LIBS}
        glfw
        OpenGL::GL
        imgui
        nfd
)

# Install rules / Portable setup (Copy OpenCV runtime dlls and AI models)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "F:/MovieAPP/HDRemaster/opencv_dev/opencv/build/x64/vc16/bin/opencv_world4110.dll"
    "F:/MovieAPP/HDRemaster/opencv_dev/opencv/build/x64/vc16/bin/opencv_videoio_ffmpeg4110_64.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/models"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/models"
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

