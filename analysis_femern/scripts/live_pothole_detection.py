#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, NavSatFix

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

        self.RosInterfaces()
        
        self.rate = rospy.Rate(5)
        rospy.loginfo("pothole deterctor started correctly")
        # self.get_single_image()

    def RosInterfaces(self):
        rospy.Subscriber("/fix", NavSatFix, self.localization_cb)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.image_depth_cb)
    
    def localization_cb(self, navSatFix_msg_:NavSatFix):
        rospy.loginfo_throttle(1, f"localization: {round(navSatFix_msg_.latitude,4)},{round(navSatFix_msg_.longitude,4)}")

    def image_depth_cb(self, image_msg_):
        pass


ros_node = RosNode()

while not rospy.is_shutdown():
    rospy.spin()