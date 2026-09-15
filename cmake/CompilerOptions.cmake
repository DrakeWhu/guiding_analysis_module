# Project-wide compiler settings, exposed through an interface target that every
# first-party target links PRIVATE (third-party code keeps its own flags).

add_library(guiding_compile_options INTERFACE)
add_library(guiding::compile_options ALIAS guiding_compile_options)

if(MSVC)
  target_compile_options(guiding_compile_options INTERFACE /W4 /permissive- /utf-8 /fp:precise)
else()
  target_compile_options(guiding_compile_options INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align -Wimplicit-fallthrough
    # numpy never fuses a*b+c into FMA; keep reductions within the parity contract.
    -ffp-contract=off)
endif()

if(GUIDING_NATIVE_ARCH AND NOT MSVC)
  target_compile_options(guiding_compile_options INTERFACE -march=native)
endif()

if(GUIDING_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "GUIDING_SANITIZE is only supported with GCC and Clang")
  endif()
  string(REPLACE ";" "," _guiding_sanitizers "${GUIDING_SANITIZE}")
  target_compile_options(guiding_compile_options INTERFACE
    -fsanitize=${_guiding_sanitizers} -fno-omit-frame-pointer)
  target_link_options(guiding_compile_options INTERFACE -fsanitize=${_guiding_sanitizers})
endif()

if(GUIDING_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _guiding_ipo OUTPUT _guiding_ipo_message)
  if(_guiding_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "GUIDING_LTO requested but unsupported: ${_guiding_ipo_message}")
  endif()
endif()

find_package(Threads REQUIRED)
