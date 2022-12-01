## code to run replays
roslaunch analysis:
```bash
roslaunch analysis_femern replay.launch bagfile:=060115_9Gb.bag
```


# conversion to pointcloud form depth image
http://wiki.ros.org/depth_image_proc#depth_image_proc.2Fpoint_cloud_xyz


# pointcloud to numpy op1
https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py