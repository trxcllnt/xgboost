# Tries to find RMM headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(RMM)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  RMM_ROOT -  When set, this path is inspected instead of standard library
#              locations as the root of the RMM installation.
#              The environment variable RMM_ROOT overrides this variable.
#
# This module defines
#  RMM_FOUND, whether RMM has been found
#  RMM_INCLUDE_DIR, directory containing header
#
# This module assumes that the user has already called find_package(CUDA)


find_path(RMM_INCLUDE_DIR
  NAMES rmm/rmm.h
  PATHS $ENV{RMM_ROOT}/include ${RMM_ROOT}/include ${CUDA_INCLUDE_DIRS} /usr/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RMM DEFAULT_MSG
                                  RMM_INCLUDE_DIR)

mark_as_advanced(
  RMM_INCLUDE_DIR
)
