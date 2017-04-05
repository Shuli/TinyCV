#!/bin/bash

# ------------------------------------
# Arg1, Processing resource type
#    CPU (Using CPU)
#    CUDA (Using CUDA)
#    X5 (Using IMP for H3)
#    CL (Using OpenCL for PowerVR)
#
# Arg2, Processing Type
#     1 (Simple dehaze removal using sample1.png. Result is printed single window)
#     2 (Measure processing, result is written on cpp_measure.csv)
#     3 (Video processing using wow.mpg)
#
# Arg3, Screen rate
#     0.5 Half screen size
#
# Arg4, Refresh number
#     5 Printing 5-step (1-step refresh rate is 1/30 sec)
#
# Other, Window-system type
#     -platform wayland (Using wayland)
# ------------------------------------

./Dehaze CPU 3 0.5 5 -platform wayland

