cmake_minimum_required(VERSION 3.4)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


project(anomalies-detecting CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()


if(MSVC)
  add_definitions(-D_WIN32_WINNT=0x600)
endif()

if(UNIX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Requires OpenCV
FIND_PACKAGE( OpenCV  REQUIRED )



add_executable(anomalies-detecting
               anomalies-detecting.cpp  main.cpp)
target_include_directories(anomalies-detecting PRIVATE )
target_link_libraries(anomalies-detecting
                      ${OpenCV_LIBRARIES})


if(WIN32 OR APPLE)
	set(CMAKE_INSTALL_PREFIX 
		"${CMAKE_BINARY_DIR}/bin"
	)
endif(WIN32 OR APPLE)

if (APPLE)
	set(BINARY_INSTALL_DIR .)
elseif (WIN32)
	set(BINARY_INSTALL_DIR .)
else ()
	set(BINARY_INSTALL_DIR bin)
endif()

install(TARGETS anomalies-detecting DESTINATION ${BINARY_INSTALL_DIR}) 


if(WIN32)
	set(BINARY_FILE "${CMAKE_INSTALL_PREFIX}/anomalies-detecting.exe")
	get_filename_component(CMAKE_TOOLCHAIN_DIRECTORY ${CMAKE_TOOLCHAIN_FILE} DIRECTORY)

	install(CODE "execute_process(COMMAND \"powershell\" \"-noprofile\" \"-executionpolicy\" \"Bypass\" \"-file\" \"${CMAKE_TOOLCHAIN_DIRECTORY}/msbuild/applocal.ps1\" \"-targetBinary\" \"${BINARY_FILE}\" \"-installedDir\" \"${CMAKE_TOOLCHAIN_DIRECTORY}/../../installed/x86-windows/bin\" \"-OutVariable\" \"out\")")

	# Install CRT
	set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)
	#if(DEVELOPER_FEATURES)
	#	set(CMAKE_INSTALL_DEBUG_LIBRARIES ON)
	#endif(DEVELOPER_FEATURES)
	set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
	include (InstallRequiredSystemLibraries)
endif(WIN32)
