#!/bin/bash

# NOTE: the old way to install python interface:
# pip install nvidia-cutlass
# then you can have three submodules to import individually:
# import cutlass_cppgen
# import cutlass_library
# import pycute

# the new way to install python interface:
pip install nvidia-cutlass-dsl
# then you can directly import:
# import cutlass