CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(FW)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(EIGEN_INCLUDE_DIR "" CACHE PATH "")
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# TODO: check the following rpath related stuff on OS X
SET(CMAKE_MACOSX_RPATH 1)
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(CMAKE_VERBOSE_MAKEFILE 1)

SET(QPBO_DIR QPBO-v1.4.src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/${QPBO_DIR})
SET(QPBO_SRCS ${QPBO_DIR}/QPBO.cpp
              ${QPBO_DIR}/QPBO_extra.cpp
              ${QPBO_DIR}/QPBO_maxflow.cpp
              ${QPBO_DIR}/QPBO_postprocessing.cpp)
SET(QPBO_HDRS ${QPBO_DIR}/QPBO.h)

SET(BUILD_SHARED_LIBS True)
SET(CMAKE_BUILD_TYPE Release)

# SET(CMAKE_INSTALL_PREFIX ..)

ADD_LIBRARY(FW SHARED
            FW.cpp
            fibonacci.h
            fibonacci.cpp
            image.h
            ${QPBO_HDRS}
            ${QPBO_SRCS})
install(TARGETS FW
        LIBRARY DESTINATION bin/Release)

# Uncomment the following lines if building a static library too is desired
# ADD_LIBRARY(FWstatic STATIC
#             FW.cpp
#             fibonacci.h
#             fibonacci.cpp
#             image.h
#             ${QPBO_HDRS}
#             ${QPBO_SRCS})
# install(TARGETS FW FWstatic
#         ARCHIVE DESTINATION bin/Release
#         LIBRARY DESTINATION bin/Release)

