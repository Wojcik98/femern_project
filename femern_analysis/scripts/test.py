#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

class RosNode:
    def __init__(self):
        rospy.init_node("ros_node")
        rospy.loginfo("Starting RosNode.")
        
        self.rate = rospy.Rate(2)
        self.get_single_image()

    def RosInterfaces(self):
        sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_raw_cb)
    
    def image_raw_cb(self, image_msg_):
        pass

    def get_single_image(self):
        _single_img:Image = rospy.wait_for_message("/camera/color/image_raw", Image)

        print(type(_single_img.data))
    
    def main_loop(self):
        self.rate.sleep()


ros_node = RosNode()

while not rospy.is_shutdown():
    ros_node.main_loop()