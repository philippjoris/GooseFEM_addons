# FindPETSc.cmake
# ----------------
# Minimal PETSc detection for custom installations
# Adjust PETSC_DIR below or pass it via -DPETSC_DIR=/path/to/petsc

find_path(PETSC_INCLUDE_DIR
    NAMES petsc.h
    PATHS
        /home/20250672/petsc-install/include
        ${PETSC_DIR}/include
        ENV PETSC_DIR
    NO_DEFAULT_PATH
)

find_library(PETSC_LIBRARY
    NAMES petsc
    PATHS
        /home/20250672/petsc-install/lib
        ${PETSC_DIR}/lib
        ENV PETSC_DIR
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
    REQUIRED_VARS PETSC_LIBRARY PETSC_INCLUDE_DIR
    FAIL_MESSAGE "Could not find PETSc. Set PETSC_DIR to your installation path."
)

if (PETSc_FOUND)
    add_library(PETSc::PETSc UNKNOWN IMPORTED)
    set_target_properties(PETSc::PETSc PROPERTIES
        IMPORTED_LOCATION "${PETSC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR}"
    )

    message(STATUS "Found PETSc include dir: ${PETSC_INCLUDE_DIR}")
    message(STATUS "Found PETSc library: ${PETSC_LIBRARY}")
endif()