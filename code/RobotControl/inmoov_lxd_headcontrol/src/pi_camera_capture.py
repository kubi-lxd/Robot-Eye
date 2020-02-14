#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import picamera
from inmoov_lxd_headcontrol.msg import head_control

class camera_capture():
    def callback(self,data):
        self.leftright = data.head_leftright
        self.updown = data.head_updown
        self.lefteye = data.head_lefteye
        self.righteye = data.head_righteye
        self.finish = data.head_finish
        if self.finish == 1:
            rospy.loginfo('head control experiments done')
        else:
	    rospy.sleep(2.5)
            self.camera.capture('/home/ras/img3/image_%d_%d_%d.bmp'%(self.leftright,self.updown,self.lefteye))
            rospy.loginfo('camera capture image_%d_%d_%d.bmp'%(self.leftright,self.updown,self.lefteye))
        
    def __init__(self):

        self.camera = picamera.PiCamera()
        self.camera.resolution = (1920,1080)
        self.camera.start_preview()
        rospy.sleep(3)
        rospy.loginfo('camera init success')
        rospy.init_node('picamera_capture_node', anonymous=False)
        rospy.Subscriber("piserial_send_data", head_control, self.callback)
        rospy.spin()

if __name__ == '__main__':
    camera = camera_capture()
