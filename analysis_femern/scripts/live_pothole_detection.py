#!/usr/bin/env python
import rospy
import cv2
import open3d as o3d
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, NavSatFix, PointCloud2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

from std_srvs.srv import Trigger

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

BASE_PATH = "/home/ionybwr/DTU/robotics_construction/femern_ws/src/femern_project/analysis_femern/"
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
        self.RosInterfaces()
        
        rospy.loginfo("pothole detector started correctly")
        rospy.logwarn("Camera instrinsics hardcoded")

    def RosInterfaces(self):
        rospy.Subscriber("/fix", NavSatFix, self.localization_cb)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.image_pointCloud_cb, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_image_cb, queue_size=1)

        self.stop_service = rospy.Service('plot_pointcloud', Trigger, self.plot_pointcloud_service,)
    
    def localization_cb(self, navSatFix_msg_:NavSatFix):
        # rospy.loginfo_throttle(1, f"localization: {round(navSatFix_msg_.latitude,4)},{round(navSatFix_msg_.longitude,4)}")
        pass

    def depth_image_cb(self, img_msg_:Image):
        return
        try:
            _cv_image = self.bridge.imgmsg_to_cv2(img_msg_, img_msg_.encoding)
        except CvBridgeError as e:
            print(e)
        
        # scaling 0-1
        _cv_image = amia.scale_img(_cv_image)
        
        cv2.imshow(f"Depth image window", _cv_image)
        cv2.waitKey(3)

    def image_pointCloud_cb(self, pc_:PointCloud2):
        _f = 5
        ps = np.array([p for i, p in enumerate(point_cloud2.read_points(pc_, skip_nans=True)) if i%_f==0])
        
        pcd_ = o3d.geometry.PointCloud()
        pcd_.points = o3d.utility.Vector3dVector(ps)

        start_ = time.time()
        if detect_pothole(pcd_):
            self.save_pointcloud_plot(pc_=pc_, fname_=f'pc_{datetime.now().time()}')
            rospy.logwarn(f"Pothole detected. Time elapsed: {time.time()-start_}")

    def save_pointcloud_plot(self, pc_:PointCloud2, fname_:str):
        """
        get pcLPointCloud2
        to numpy (reduced by _f)
        convert to o3d
        voxel-down sample
        make and save html interactive
        """
        _f = 5 # one point out of _f 
        ps = np.array([p for i, p in enumerate(point_cloud2.read_points(pc_, skip_nans=True)) if i%_f==0])

        # pcd_ = o3d.geometry.PointCloud()
        # pcd_.points = o3d.utility.Vector3dVector(ps)
        
        fig = px.scatter_3d(x=ps[:,0].tolist(), y=ps[:,1].tolist(), z=ps[:,2].tolist())
        fig.update_traces(marker=dict(size=3))
        fig.write_html(BASE_PATH+f'{fname_}.html')

        rospy.loginfo("pointcloud image saved")

    def plot_pointcloud_service(self, req):
        _pc = rospy.wait_for_message("/camera/depth/points", PointCloud2)
        
        return_msg = 'Trigger plot service called'
        return_bool = False

        _start = time.time()
        _f = 5
        ps = np.array([p for i, p in enumerate(point_cloud2.read_points(_pc, skip_nans=True)) if i%_f==0])
        _comprehension_making_time = time.time() - _start

        _start = time.time()

        fig = px.scatter_3d(x=ps[:,0].tolist(), y=ps[:,1].tolist(), z=ps[:,2].tolist())
        fig.update_traces(marker=dict(size=3))
        fig.write_html(BASE_PATH+f'service_pc_{datetime.now().time()}.html')
        _fig_making_time = time.time() - _start

        return_msg = f"reduced points: {ps.shape} f:{_f} {_comprehension_making_time=} {_fig_making_time=}"
        return_bool = True

        return return_bool, return_msg
    
    def plot_figure(self):
        if self.fig is not None:
            plt.show()
            self.fig = None

ros_node = RosNode()

while not rospy.is_shutdown():
    try:
        rospy.spin()
        ros_node.plot_figure()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()