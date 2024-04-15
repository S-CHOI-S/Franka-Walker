cmake_minimum_required(VERSION 3.15)
project(simulation)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")

#fing eigen library
find_package (Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
#find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED) # pybind11

# set(PYTHON_INCLUDE_DIR "/home/kist/anaconda3/envs/mujoco_gym/include/python3.8")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-2.3.3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-2.3.3/simulate
  $ENV{EIGEN3_INCLUDE_DIR}/Eigen/
  ${GLEW_INCLUDE_DIRS}
    /usr/local/include/rbdl
    /usr/local/lib
    ${PYTHON_INCLUDE_DIR}
)

set(COM_SRC
    src/main.cc
    ${MUJOCO_HOME}/simulate/glfw_dispatch.cc
    ${MUJOCO_HOME}/simulate/glfw_adapter.cc
    ${MUJOCO_HOME}/simulate/simulate.cc
    ${MUJOCO_HOME}/simulate/platform_ui_adapter.cc    
    
        
        src/controller.cpp
        src/robotmodel.cpp
        src/trajectory.cpp
        src/quadraticprogram.cpp
        
#       src/pybind_controller.cpp
#       example.cpp
        )

set(COM_LIB 
${MUJOCO_HOME}/lib/libmujoco.so
${MUJOCO_HOME}/lib/libmujoco.so.2.3.3
${MUJOCO_HOME}/bin/mujoco_plugin/libelasticity.so
${GLFW}
-lqpOASES
)

add_executable(mppi ${COM_SRC})

target_link_libraries(mppi 
	${COM_LIB} 
	${CMAKE_DL_LIBS} 
	${X11_LIBRARIES}
	-lrbdl
    	-lrbdl_urdfreader
    	-lpthread
    	OpenGL::OpenGL
	OpenGL::EGL
	GLEW::GLEW
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
)

# pybind11
set_source_files_properties(${COM_SRC} PROPERTIES COMPILE_FLAGS "-fPIC")

add_library(pybind_lib
	${COM_SRC}
)

pybind11_add_module(mjc_controller src/pybind_controller.cpp)
target_link_libraries(mjc_controller PUBLIC
	Eigen3::Eigen
	pybind_lib
	-lrbdl
	-lrbdl_urdfreader
)

pybind11_add_module(example example.cpp)
target_link_libraries(example PUBLIC
	Eigen3::Eigen
	pybind_lib
	-lrbdl
	-lrbdl_urdfreader
)

