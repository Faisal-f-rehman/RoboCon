cmake_minimum_required(VERSION 3.5)

project(RoboCon LANGUAGES CXX)

find_package( OpenCV REQUIRED )
find_package(Boost 1.58.0 REQUIRED COMPONENTS system filesystem)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wno-write-strings" )
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(RoboCon main.cpp vision.cpp guicomms.cpp robosocket.cpp)
#add_executable(RoboCon main.cpp vision.cpp guicomms.cpp)

target_link_libraries(RoboCon ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_REGEX_LIBRARY} )


#set(CMAKE_CXX_STANDARD 11)
#find_package(Boost 1.40.0 REQUIRED system)


