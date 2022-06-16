function roscore() {
    ssh nvidia@192.168.0.105 <<EOF
    roscore
EOF
}

function pi_bringup() {
    ssh ubuntu@192.168.0.100 roslaunch turtlebot3_bringup turtlebot3_robot.launch
}

function ros_deep_learning() {
    ssh nvidia@192.168.0.105 roslaunch ros_deep_learning video_source.ros1.launch input:=csi://0
}

function converter() {
    ssh nvidia@192.168.0.105 python converter.py
}

function lidar_scan() {
    ssh nvidia@192.168.0.105 python lidar_scan_heading.py
}

function darknet() {
    ssh nvidia@192.168.0.105 roslaunch darknet_ros darknet_ros.launch
}

function bringup() {
    ssh nvidia@192.168.0.105 roslaunch turtlebot3_manipulation_bringup turtlebot3_manipulation_bringup.launch
}

function moveit() {
    ssh nvidia@192.168.0.105 roslaunch turtlebot3_manipulation_moveit_config move_group.launch
}

function run() {
    ssh nvidia@192.168.0.105 cd robocats-final;python run.py $1
}