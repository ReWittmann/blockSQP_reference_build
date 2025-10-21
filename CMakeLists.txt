cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(original_blockSQP_interface    
	DESCRIPTION "Build system for blockSQP "
	HOMEPAGE_URL https://github.com/djanka2/blockSQP
    LANGUAGES CXX C Fortran
)
set(CMAKE_CXX_STANDARD 11)
#enable_language(Fortran)


#Dependencies: LAPACK, BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS_FOUND = ${BLAS_FOUND}")
message(STATUS "BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
message(STATUS "BLAS_LINKER_FLAGS = ${BLAS_LINKER_FLAGS}")
message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")

set(BLAS_LIBRARY ${BLAS_LIBRARIES})
set(LAPACK_LIBRARY ${LAPACK_LIBRARIES})


include(FetchContent)

#Dependency: MUMPS sparse linear solver
FetchContent_Declare(
	CMAKEMUMPS
	GIT_REPOSITORY https://github.com/scivision/mumps.git
	GIT_TAG 5b153379cdd1b0d2a33daf88bffab16a22ff0004
)
set(MUMPS_parallel OFF)
set(MUMPS_scalapack OFF)
set(BUILD_SINGLE OFF)
set(MUMPS_find_static OFF)
set(MUMPS_intsize64 OFF) #must be off
set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} -Os) #Likely to be loaded several times with dlmopen + no noticeable performance difference compared to -O3
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(CMAKEMUMPS)

#Dependency: qpOASES. Apply patch to fix MUMPS interface
FetchContent_Declare(
	qpos
	GIT_REPOSITORY	https://github.com/coin-or/qpOASES.git
	GIT_TAG			35b762ba3fee2e009d9e99650c68514da05585c5
	SOURCE_SUBDIR 	DIR/THAT/DOESNT/EXIST
	PATCH_COMMAND	git apply ${CMAKE_CURRENT_SOURCE_DIR}/qpOASES_fix_MUMPS.patch
	UPDATE_DISCONNECTED 1
)
FetchContent_MakeAvailable(qpos)

add_library(qpOASES STATIC
	${qpos_SOURCE_DIR}/src/Matrices.cpp          
	${qpos_SOURCE_DIR}/src/SparseSolver.cpp
	${qpos_SOURCE_DIR}/src/Bounds.cpp             
	${qpos_SOURCE_DIR}/src/MessageHandling.cpp   
	${qpos_SOURCE_DIR}/src/SQProblem.cpp
	${qpos_SOURCE_DIR}/src/Constraints.cpp        
	${qpos_SOURCE_DIR}/src/Options.cpp           
	${qpos_SOURCE_DIR}/src/SQProblemSchur.cpp
	${qpos_SOURCE_DIR}/src/Flipper.cpp            
	${qpos_SOURCE_DIR}/src/OQPinterface.cpp      
	${qpos_SOURCE_DIR}/src/SubjectTo.cpp
	${qpos_SOURCE_DIR}/src/Indexlist.cpp          
	${qpos_SOURCE_DIR}/src/QProblemB.cpp      
	${qpos_SOURCE_DIR}/src/QProblemB.cpp         
	${qpos_SOURCE_DIR}/src/Utils.cpp
	${qpos_SOURCE_DIR}/src/QProblem.cpp     
	${qpos_SOURCE_DIR}/src/QProblem.cpp           
	${qpos_SOURCE_DIR}/src/SolutionAnalysis.cpp
)
target_compile_definitions(qpOASES PUBLIC SOLVER_MUMPS PRIVATE MUMPS_SEQ __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3 INTERFACE -Wno-overloaded-virtual PUBLIC -fPIC)
target_link_libraries(qpOASES PUBLIC MUMPS::MUMPS)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
		target_link_libraries(qpOASES PUBLIC -lgfortran)
ELSEIF(WIN32)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpos_SOURCE_DIR}/include
			${qpos_SOURCE_DIR}/include/qpOASES
)

#Fetch and build blockSQP
FetchContent_Declare(
	bsqp
	GIT_REPOSITORY	https://github.com/ReWittmann/blockSQP_reference
	GIT_TAG			da05d3957b93bd55b9a7f08c859d302851056a6d
	SOURCE_SUBDIR 	DIR/THAT/DOESNT/EXIST
	PATCH_COMMAND	git apply ${CMAKE_CURRENT_SOURCE_DIR}/blockSQP_fix_doublefree_oobwrite.patch
	UPDATE_DISCONNECTED 1
)
FetchContent_MakeAvailable(bsqp)

add_library(blockSQP STATIC
	${bsqp_SOURCE_DIR}/src/blocksqp_general_purpose.cpp
	${bsqp_SOURCE_DIR}/src/blocksqp_glob.cpp
   	${bsqp_SOURCE_DIR}/src/blocksqp_hess.cpp
   	${bsqp_SOURCE_DIR}/src/blocksqp_iter.cpp
   	${bsqp_SOURCE_DIR}/src/blocksqp_main.cpp
	${bsqp_SOURCE_DIR}/src/blocksqp_matrix.cpp
	${bsqp_SOURCE_DIR}/src/blocksqp_options.cpp
	${bsqp_SOURCE_DIR}/src/blocksqp_problemspec.cpp
   	${bsqp_SOURCE_DIR}/src/blocksqp_qp.cpp
   	${bsqp_SOURCE_DIR}/src/blocksqp_restoration.cpp
   	${bsqp_SOURCE_DIR}/src/blocksqp_stats.cpp
   	)

target_compile_options(blockSQP PRIVATE 
	-O3
	-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized)

set_target_properties(blockSQP PROPERTIES 
						LIBRARY_OUTPUT_DIRECTORY ${bsqp_SOURCE_DIR}/bin
						ARCHIVE_OUTPUT_DIRECTORY ${bsqp_SOURCE_DIR}/lib
						RUNTIME_OUTPUT_DIRECTORY ${bsqp_SOURCE_DIR}/bin)

target_include_directories(blockSQP PUBLIC ${bsqp_SOURCE_DIR}/include)
target_link_libraries(blockSQP PUBLIC qpOASES)

#Build blockSQP example
add_executable(example1 ${bsqp_SOURCE_DIR}/examples/example1.cpp)
target_compile_options(example1 PRIVATE -O3)
target_link_libraries(example1 blockSQP)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${bsqp_SOURCE_DIR}/examples")


#Python interface for blockSQP
set(PYTHON_INTERPRETER "" CACHE STRING "The python interpreter the interface should be built for, passes value to the PYTHON_EXECUTABLE variable used by pybind11")
if (NOT PYTHON_INTERPRETER STREQUAL "")
	message(STATUS "Using python interpreter ${PYTHON_INTERPRETER}")
	set (PYTHON_EXECUTABLE ${PYTHON_INTERPRETER})
endif()

FetchContent_Declare(
	PYBIND11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4	#2.13.6
)
FetchContent_MakeAvailable(PYBIND11)

pybind11_add_module(py_blockSQP ${CMAKE_CURRENT_SOURCE_DIR}/py_blockSQP.cpp)
#target_include_directories(py_blockSQP PUBLIC ${bsqp_SOURCE_DIR}/include)	

if(UNIX)
	target_compile_options(py_blockSQP PRIVATE 
				-Wall -Wextra -Wno-unused-parameter
				-fvisibility=hidden
				-O3
				)
	
	#Prevent warnings for *_pointer_interface and T_array constructors 
	#that occur after compiling with -O3. Result of using int instead of size_t, should not cause any problems.
	target_link_options(py_blockSQP PRIVATE -Wno-alloc-size-larger-than)
endif()

target_link_libraries(py_blockSQP PUBLIC blockSQP)

set(py_blockSQP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/py_blockSQP_old)
set_target_properties(py_blockSQP PROPERTIES 
						LIBRARY_OUTPUT_DIRECTORY ${py_blockSQP_DIR}
						RUNTIME_OUTPUT_DIRECTORY ${py_blockSQP_DIR}
						)


