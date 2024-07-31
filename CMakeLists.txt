cmake_minimum_required(VERSION 3.15)
project(simulation_walker)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")

find_package (Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(pybind11 REQUIRED) # pybind11

message(STATUS "Eigen3 : " ${EIGEN3_INCLUDE_DIR}/Eigen)
message(STATUS "current : " ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DMJSIMULATE_STATIC) # use static =,solve the "error LNK2001/LNK2019" in windows

find_package (X11)
find_library(GLFW libglfw.so.3 HINTS ${CMAKE_SOURCE_DIR}/mujoco200/bin)
# set MUJOCO root
set(MUJOCO_HOME ${CMAKE_SOURCE_DIR}/mujoco-2.3.3/)
message(STATUS "MUJOCO_HOME: " ${MUJOCO_HOME})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIRS}
)

#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/
)

#$ENV{EIGEN3_INCLUDE_DIR}

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-2.3.3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-2.3.3/simulate
  $ENV{EIGEN3_INCLUDE_DIR}/Eigen/
  ${GLEW_INCLUDE_DIRS}
    /usr/local/include/rbdl
    /usr/local/lib
    ${PYTHON_INCLUDE_DIR}
)

set(COM_SRC
    ${MUJOCO_HOME}/simulate/glfw_dispatch.cc
    ${MUJOCO_HOME}/simulate/glfw_adapter.cc
    ${MUJOCO_HOME}/simulate/simulate.cc
    ${MUJOCO_HOME}/simulate/platform_ui_adapter.cc    
   )

set(COM_LIB 
${MUJOCO_HOME}/lib/libmujoco.so
${MUJOCO_HOME}/lib/libmujoco.so.2.3.3
${MUJOCO_HOME}/bin/mujoco_plugin/libelasticity.so
${GLFW}
-lqpOASES
)

# pybind11
set_source_files_properties(${COM_SRC} PROPERTIES COMPILE_FLAGS "-fPIC")

add_library(pybind_lib
	${COM_SRC}
)


