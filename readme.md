# original_blockSQP_build - build system for blockSQP
This project contains CMake build specifications, patches, a python interface and example problems for the blockSQP release by Dennis Janka <https://github.com/djanka2/blockSQP>

Copyright (c) 2025- Reinhold Wittmann <reinhold.wittmann@ovgu.de>

##Build requirements
1. Linux
2. CMake
3. Git
4. A fortran compiler, e.g. gfortran
5. A C++ 11 compiler, e.g. g++-14

##Building
Invoke CMake on the CMakeLists.txt, e.g. navigate to the folder and use the commands  
&nbsp;&nbsp; cmake -B .build  
&nbsp;&nbsp; cmake --build .build  



If you wish to select a specific python installation to build for, use  
&nbsp;&nbsp; cmake -B .build -DPYTHON_INTERPRETER=PATH/TO/PYTHON/EXECUTABLE

