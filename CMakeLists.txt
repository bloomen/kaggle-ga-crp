cmake_minimum_required(VERSION 2.8)
project(generate_submission)

find_package(Threads)

add_executable(generate_submission
	generate_submission.cpp)

if (MSVC)
	set(CMAKE_CXX_FLAGS "/W4 /bigobj /EHsc /wd4503 /wd4996 /wd4702")
else()
	set(CMAKE_CXX_FLAGS "-std=c++17 -pedantic -Wall -Wextra -Wconversion")
	if (CMAKE_COMPILER_IS_GNUCC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	endif()
	target_link_libraries(generate_submission ${CMAKE_THREAD_LIBS_INIT})
endif()
