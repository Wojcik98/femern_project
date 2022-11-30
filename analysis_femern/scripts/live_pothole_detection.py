#!/usr/bin/env python
import rospy
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, NavSatFix, PointCloud2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from dataclasses import dataclass
import time
import pyrealsense2 as rs

import extras.ImageAnalysis as amia
@dataclass
class CameraIntrinsics:
    def __init__(self, k_array) -> None:
        self.fx = k_array[0]
        self.ppx = k_array[2]
        self.fy = k_array[4]
        self.ppy = k_array[5]
        self.k = np.array(
            [[self.fx, 0 , self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]]
            )

def detect_pothole(xyz):
    plane_model, inliers = xyz.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=100)
    outlier_cloud = xyz.select_by_index(inliers, invert=True)

    pc_np = np.asarray(outlier_cloud.points)
    if not pc_np.size:
        rospy.loginfo(f"pothole detected size: {pc_np.size}, {pc_np[:10]}")
        return True

    return False

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics:CameraIntrinsics ):
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	
	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
	y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

	z = depth_image.flatten() / 1000
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]

	return x, y, z
class RosNode:
    def __init__(self):
        """
        TODO: subcribe to image pointcloud
        TODO: subcribe to gps ('/fix')
        TODO: fit plane to pointcould
        TODO: Identify whether there are potholes in the frame
        TODO: save coordinates of pothole location
        
        """
        rospy.init_node("pothole_detection_node")
        rospy.loginfo("Starting RosNode...")

        self.bridge = CvBridge()
        self.camera_instrinsics = CameraIntrinsics([387.3398132324219, 0.0, 319.65350341796875, 0.0, 387.3398132324219, 243.85708618164062, 0.0, 0.0, 1.0])
        self.cb_recieved_time = time.time()
        self.RosInterfaces()
        
        # self.rate = rospy.Rate(5)
        rospy.loginfo("pothole deterctor started correctly")
        # self.get_single_image()

    def RosInterfaces(self):
        rospy.Subscriber("/fix", NavSatFix, self.localization_cb)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.image_pointCloud_cb, queue_size=1)
    
    def localization_cb(self, navSatFix_msg_:NavSatFix):
        # rospy.loginfo_throttle(1, f"localization: {round(navSatFix_msg_.latitude,4)},{round(navSatFix_msg_.longitude,4)}")
        pass

    def image_pointCloud_cb(self, pc_:PointCloud2):
        _cb_time = time.time()-self.cb_recieved_time
        # rospy.loginfo(f"msg recieved. time from last: {_cb_time}")
        self.cb_recieved_time = time.time()
        try:
            _cv_image = self.bridge.imgmsg_to_cv2(pc_, pc_.encoding)
        except CvBridgeError as e:
            print(e)
        
        # scaling 0-1
        _cv_image = amia.scale_img(_cv_image)
        # amia.img_stats(_cv_image,print_=1)
        
        cv2.imshow(f"Depth image window", _cv_image)
        cv2.waitKey(3)

        # conversion to pointcloud

        x, y, z = convert_depth_frame_to_pointcloud(_cv_image, self.camera_instrinsics)

        xyz = np.stack((x,y,z),axis=1)
        pcd_ = o3d.geometry.PointCloud()
        pcd_.points = o3d.utility.Vector3dVector(xyz)

        start_ = time.time()
        if detect_pothole(pcd_):
        # if True:
            stop_ = time.time()
            rospy.logwarn(f"Pothole detected. Time elapsed: {stop_-start_}")

ros_node = RosNode()

while not rospy.is_shutdown():
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()