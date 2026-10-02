#!/bin/bash
source install/setup.bash
ros2 run rb2301_ca1 obstacle_avoidance 2>&1 | tee out.txt
