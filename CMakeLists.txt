cmake_minimum_required(VERSION 3.20)
project(Delve)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
link_directories(lib)

# ── Game ──────────────────────────────────────────────────────────────────────
add_executable(Delve src/main.cpp)
target_link_libraries(Delve raylib opengl32 gdi32 winmm)

# ── Launcher ──────────────────────────────────────────────────────────────────
# Links winhttp  (built-in Windows HTTPS client — no extra DLLs needed)
# Links shell32  (ShellExecuteA for running the installer with UAC elevation)
# -mwindows suppresses the console window on launch
add_executable(DelveLauncher WIN32 src/launcher.cpp)
target_link_libraries(DelveLauncher raylib opengl32 gdi32 winmm winhttp shell32)
target_link_options(DelveLauncher PRIVATE -mwindows)