cmake_minimum_required (VERSION 2.6)

project (AdaBoost)

set (CMAKE_BUILD_TYPE Release)

add_executable(abtrain abtrain.cpp readSampleDataFile.cpp AdaBoost.cpp)
add_executable(abpredict abpredict.cpp readSampleDataFile.cpp AdaBoost.cpp)
