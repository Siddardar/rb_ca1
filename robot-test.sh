#!/usr/bin/env bash

set -e

# Usage:
#   ./robot.sh 02
#
# 02 -> 192.168.1.202
# 15 -> 192.168.1.215

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]{2}$ ]]; then
    echo "Usage: $0 <2-digit robot number>"
    echo "Example: $0 02"
    exit 1
fi

ROBOT_NUM="$1"
ROBOT_IP="192.168.1.2${ROBOT_NUM}"
ROBOT_USER="bingda"
ROBOT_PASS="bingda"

LOCAL_SRC="$HOME/Projects/rb2301_ca1/src/rb2301_ca1"

REMOTE_WS="/home/bingda/Downloads/rb2301_ca1-11"
REMOTE_SRC="$REMOTE_WS/src"

# Shared control files on robot
START_FILE="/tmp/rb2301_obstacle.start"
PID_FILE="/tmp/rb2301_obstacle.pid"
QUIT_FILE="/tmp/rb2301_quit"

SSH="sshpass -p ${ROBOT_PASS} ssh -o StrictHostKeyChecking=no"
RSYNC_RSH="sshpass -p ${ROBOT_PASS} ssh -o StrictHostKeyChecking=no"


echo "================================"
echo "Robot: BINGDA-0${ROBOT_NUM}"
echo "IP:    ${ROBOT_IP}"
echo "================================"


# ==================================================
# 1. Create workspace
# ==================================================

echo "Creating workspace..."

$SSH "${ROBOT_USER}@${ROBOT_IP}" \
    "mkdir -p ${REMOTE_SRC}"


# ==================================================
# 2. Sync source
# ==================================================

echo "Syncing source..."

rsync -auvx --delete \
    -e "$RSYNC_RSH" \
    "$LOCAL_SRC" \
    "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_SRC}"


# ==================================================
# 3. Build
# ==================================================

echo "Building workspace..."

$SSH "${ROBOT_USER}@${ROBOT_IP}" "
    cd ${REMOTE_WS} &&
    colcon build --symlink-install
"


# ==================================================
# 4. Remove old control files
# ==================================================

$SSH "${ROBOT_USER}@${ROBOT_IP}" "
    rm -f ${START_FILE}
    rm -f ${PID_FILE}
    rm -f ${QUIT_FILE}
"


echo "Build complete."
echo "Opening terminals..."


# ==================================================
# TERMINAL 1 — BASE CONTROL
# ==================================================

gnome-terminal \
    --title="BINGDA Base Control" \
    -- bash -c "
        sshpass -p '${ROBOT_PASS}' ssh \
        -o StrictHostKeyChecking=no \
        -tt ${ROBOT_USER}@${ROBOT_IP} \
        \"bash -c '

            cd ${REMOTE_WS}
            source install/setup.bash

            echo \\\"Starting Base Control...\\\"

            setsid ros2 launch \
                base_control_ros2 \
                base_control.launch.py \
                </dev/null &

            PID=\\\$!

            while kill -0 \\\$PID 2>/dev/null; do

                if [ -f ${QUIT_FILE} ]; then
                    echo
                    echo \\\"[GLOBAL SHUTDOWN]\\\"

                    kill -INT -- -\\\$PID 2>/dev/null || true
                    wait \\\$PID 2>/dev/null || true

                    break
                fi

                sleep 0.2

            done

        '\"
    "


# ==================================================
# TERMINAL 2 — LIDAR
# ==================================================

gnome-terminal \
    --title="BINGDA LiDAR" \
    -- bash -c "
        sshpass -p '${ROBOT_PASS}' ssh \
        -o StrictHostKeyChecking=no \
        -tt ${ROBOT_USER}@${ROBOT_IP} \
        \"bash -c '

            cd ${REMOTE_WS}
            source install/setup.bash

            echo \\\"Starting LiDAR...\\\"

            setsid ros2 launch \
                rplidar_ros \
                rplidar.launch.py \
                </dev/null &

            PID=\\\$!

            while kill -0 \\\$PID 2>/dev/null; do

                if [ -f ${QUIT_FILE} ]; then
                    echo
                    echo \\\"[GLOBAL SHUTDOWN]\\\"

                    kill -INT -- -\\\$PID 2>/dev/null || true
                    wait \\\$PID 2>/dev/null || true

                    break
                fi

                sleep 0.2

            done

        '\"
    "


# ==================================================
# TERMINAL 3 — OBSTACLE AVOIDANCE OUTPUT
# ==================================================

gnome-terminal \
    --title="Obstacle Avoidance OUTPUT" \
    -- bash -c "
        sshpass -p '${ROBOT_PASS}' ssh \
        -o StrictHostKeyChecking=no \
        -tt ${ROBOT_USER}@${ROBOT_IP} \
        \"bash -c '

            cd ${REMOTE_WS}
            source install/setup.bash

            echo
            echo \\\"================================\\\"
            echo \\\" Obstacle Avoidance OUTPUT\\\"
            echo \\\"================================\\\"
            echo
            echo \\\"Waiting for START...\\\"

            # Wait until control terminal starts us
            while [ ! -f ${START_FILE} ]; do

                # Allow global shutdown even before start
                if [ -f ${QUIT_FILE} ]; then
                    exit 0
                fi

                sleep 0.1

            done

            rm -f ${START_FILE}

            echo
            echo \\\"Starting obstacle avoidance...\\\"
            echo

            setsid ros2 run \
                rb2301_ca1 \
                obstacle_avoidance \
                </dev/null &

            PID=\\\$!

            echo \\\$PID > ${PID_FILE}

            echo \\\"PID: \\\$PID\\\"
            echo \\\"--------------------------------\\\"
            echo

            while kill -0 \\\$PID 2>/dev/null; do

                if [ -f ${QUIT_FILE} ]; then

                    echo
                    echo \\\"[GLOBAL SHUTDOWN]\\\"

                    # Make sure paused process is resumed
                    # so it can receive SIGINT.
                    kill -CONT -- -\\\$PID 2>/dev/null || true

                    kill -INT -- -\\\$PID 2>/dev/null || true

                    wait \\\$PID 2>/dev/null || true

                    break
                fi

                sleep 0.1

            done

            rm -f ${PID_FILE}

        '\"
    "


# ==================================================
# TERMINAL 4 — CONTROL
#
# ENTER -> start
# s     -> pause
# r     -> resume
# q     -> GLOBAL SHUTDOWN
# ==================================================

gnome-terminal \
    --title="Obstacle Avoidance CONTROL" \
    -- bash -c "
        sshpass -p '${ROBOT_PASS}' ssh \
        -o StrictHostKeyChecking=no \
        -tt ${ROBOT_USER}@${ROBOT_IP} \
        \"bash -c '

            echo
            echo \\\"================================\\\"
            echo \\\" Obstacle Avoidance CONTROL\\\"
            echo \\\"================================\\\"
            echo
            echo \\\"ENTER = start\\\"
            echo \\\"q     = quit everything\\\"
            echo

            read -r -p \\\"Press ENTER to START: \\\" CMD

            # q can also quit before starting
            if [ \\\"\\\$CMD\\\" = \\\"q\\\" ]; then
                touch ${QUIT_FILE}
                exit 0
            fi

            touch ${START_FILE}

            echo \\\"Waiting for obstacle process...\\\"

            while [ ! -s ${PID_FILE} ]; do

                if [ -f ${QUIT_FILE} ]; then
                    exit 0
                fi

                sleep 0.1

            done

            PID=\\\$(cat ${PID_FILE})

            echo
            echo \\\"Obstacle avoidance started.\\\"
            echo
            echo \\\"Commands:\\\"
            echo \\\"  s + ENTER  -> pause\\\"
            echo \\\"  r + ENTER  -> resume\\\"
            echo \\\"  q + ENTER  -> STOP EVERYTHING\\\"
            echo

            while true; do

                read -r -p \\\"command> \\\" CMD

                case \\\"\\\$CMD\\\" in

                    s)
                        kill -STOP -- -\\\$PID 2>/dev/null || true
                        echo \\\"[PAUSED]\\\"
                        ;;

                    r)
                        kill -CONT -- -\\\$PID 2>/dev/null || true
                        echo \\\"[RESUMED]\\\"
                        ;;

                    q)
                        echo
                        echo \\\"[GLOBAL SHUTDOWN]\\\"
                        echo \\\"Stopping all ROS processes...\\\"

                        # Signal every other terminal
                        touch ${QUIT_FILE}

                        exit 0
                        ;;

                    *)
                        echo \\\"Use s, r, or q\\\"
                        ;;

                esac

            done

        '\"
    "


echo "================================"
echo "All 4 terminals opened."
echo "================================"