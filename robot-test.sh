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

REMOTE_WS="~/Downloads/rb2301_ca1-11"
REMOTE_SRC="$REMOTE_WS/src"

PID_FILE="/tmp/rb2301_obstacle.pid"
START_FILE="/tmp/rb2301_obstacle.start"

SSH="sshpass -p ${ROBOT_PASS} ssh -o StrictHostKeyChecking=no"
RSYNC_RSH="sshpass -p ${ROBOT_PASS} ssh -o StrictHostKeyChecking=no"

echo "================================"
echo "Robot: BINGDA-0${ROBOT_NUM}"
echo "IP:    ${ROBOT_IP}"
echo "================================"

# --------------------------------------------------
# 1. Create workspace
# --------------------------------------------------

echo "Creating workspace..."

$SSH "${ROBOT_USER}@${ROBOT_IP}" \
    "mkdir -p ${REMOTE_SRC}"

# --------------------------------------------------
# 2. Sync source code
# --------------------------------------------------

echo "Syncing source..."

rsync -auvx --delete \
    -e "$RSYNC_RSH" \
    "$LOCAL_SRC" \
    "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_SRC}"

# --------------------------------------------------
# 3. Build
# --------------------------------------------------

echo "Building workspace..."

$SSH "${ROBOT_USER}@${ROBOT_IP}" "
    cd ${REMOTE_WS} &&
    colcon build --symlink-install
"

echo "Build complete."

# --------------------------------------------------
# 4. Clean old obstacle-control files
# --------------------------------------------------

$SSH "${ROBOT_USER}@${ROBOT_IP}" "
    rm -f ${PID_FILE}
    rm -f ${START_FILE}
"

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
        'cd ${REMOTE_WS} &&
         source install/setup.bash &&
         ros2 launch base_control_ros2 base_control.launch.py';

        exec bash
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
        'cd ${REMOTE_WS} &&
         source install/setup.bash &&
         ros2 launch rplidar_ros rplidar.launch.py';

        exec bash
    "

# ==================================================
# TERMINAL 3 — OBSTACLE AVOIDANCE OUTPUT
#
# This terminal waits until Terminal 4 tells it
# to start.
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
            echo \\\"Waiting for START command...\\\"

            while [ ! -f ${START_FILE} ]; do
                sleep 0.1
            done

            rm -f ${START_FILE}

            echo
            echo \\\"Starting obstacle avoidance...\\\"
            echo

            setsid ros2 run rb2301_ca1 obstacle_avoidance </dev/null &

            PID=\\\$!

            echo \\\$PID > ${PID_FILE}

            echo \\\"Process PID: \\\$PID\\\"
            echo \\\"--------------------------------\\\"
            echo

            wait \\\$PID

            rm -f ${PID_FILE}

            echo
            echo \\\"--------------------------------\\\"
            echo \\\"Obstacle avoidance stopped.\\\"
        '\"

        exec bash
    "

# ==================================================
# TERMINAL 4 — OBSTACLE AVOIDANCE CONTROL
#
# ENTER -> start
# s     -> pause
# r     -> resume
# q     -> stop
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

            read -r -p \\\"Press ENTER to START...\\\"

            touch ${START_FILE}

            echo \\\"Waiting for obstacle process...\\\"

            while [ ! -s ${PID_FILE} ]; do
                sleep 0.1
            done

            PID=\\\$(cat ${PID_FILE})

            echo
            echo \\\"Obstacle avoidance started.\\\"
            echo \\\"PID: \\\$PID\\\"
            echo
            echo \\\"Commands:\\\"
            echo \\\"  s + ENTER  -> pause\\\"
            echo \\\"  r + ENTER  -> resume\\\"
            echo \\\"  q + ENTER  -> stop\\\"
            echo

            while true; do

                read -r -p \\\"command> \\\" CMD

                case \\\"\\\$CMD\\\" in

                    s)
                        kill -STOP -- -\\\$PID

                        echo \\\"[PAUSED]\\\"
                        ;;

                    r)
                        kill -CONT -- -\\\$PID

                        echo \\\"[RESUMED]\\\"
                        ;;

                    q)
                        echo \\\"[STOPPING]\\\"

                        kill -INT -- -\\\$PID 2>/dev/null || true

                        break
                        ;;

                    *)
                        echo \\\"Use s, r, or q\\\"
                        ;;

                esac

            done

            echo \\\"Obstacle avoidance stopped.\\\"

        '\"

        exec bash
    "

echo "================================"
echo "All 4 terminals opened."
echo "================================"
