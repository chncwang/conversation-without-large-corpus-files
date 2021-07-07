CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(single-turn-conversation)

SET(CMAKE_CXX_COMPILER g++)

######## Cross-compiler, cross-platform options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH -Wall -Wno-sign-compare")
if (MKL OR MKL_ROOT)
  find_mkl()  # sets include/lib directories and sets ${LIBS} needed for linking
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
endif()

if(CMAKE_BUILD_TYPE MATCHES debug)
    ADD_DEFINITIONS(-DN3LDG_DEBUG)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -O1" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -O3 -march=native" )
endif()

SET(Boost_USE_STATIC_LIBS OFF) 
SET(Boost_USE_MULTITHREADED ON)  
SET(Boost_USE_STATIC_RUNTIME OFF) 
FIND_PACKAGE(Boost REQUIRED COMPONENTS regex system filesystem) 
MESSAGE("boost includes:" ${Boost_INCLUDE_DIRS})
MESSAGE("boost include:" ${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}) 
INCLUDE_DIRECTORIES(${EIGEN})
INCLUDE_DIRECTORIES(3rd/jsoncpp)
INCLUDE_DIRECTORIES(3rd/n3ldg/include 3rd/cxxopts/include src)
INCLUDE_DIRECTORIES(3rd/inih)
INCLUDE_DIRECTORIES(3rd/tinyutf8)
INCLUDE_DIRECTORIES(3rd/mteval)

AUX_SOURCE_DIRECTORY(3rd/jsoncpp SRCS)
AUX_SOURCE_DIRECTORY(3rd/mteval/mteval SRCS)
AUX_SOURCE_DIRECTORY(src/single_turn_conversation SRCS)
ADD_EXECUTABLE(main ${SRCS})

IF(TEST_CUDA)
    ADD_DEFINITIONS(-DTEST_CUDA)
ENDIF()

IF(USE_DOUBLE)
    ADD_DEFINITIONS(-DUSE_DOUBLE)
ENDIF()

IF(USE_CUDA)
    ADD_DEFINITIONS(-DUSE_GPU)
ENDIF()

SET(LIBS ${LIBS} insnet)
add_subdirectory(3rd/insnet)
TARGET_LINK_LIBRARIES(main pthread ${LIBS})
TARGET_LINK_LIBRARIeS(main ${Boost_LIBRARIES})
