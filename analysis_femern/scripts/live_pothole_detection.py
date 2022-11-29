#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

class RosNode:
    def __init__(self):
        """
        TODO: subcribe to image pointcloud
        TODO: subcribe to gps ('/fix')
        TODO: fit plane to pointcould
        TODO: Identify whether there are potholes in the frame
        TODO: save coordinates of pothole location
        
        """
        rospy.init_node("pothole_detection")
        # rospy.loginfo("Starting RosNode.")
        
        self.rate = rospy.Rate(2)
        self.get_single_image()

    def RosInterfaces(self):
        sub = rospy.Subscriber("/fix", Image, self.localization_cb)
        sub = rospy.Subscriber("/fix", Image, self.image_depth_cb)
    
    def image_raw_cb(self, image_msg_):
        pass
    
    def main_loop(self):
        self.rate.sleep()


ros_node = RosNode()

while not rospy.is_shutdown():
    rospy.spin()