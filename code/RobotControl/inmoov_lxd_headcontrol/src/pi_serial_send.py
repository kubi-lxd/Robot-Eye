#!/usr/bin/env python
# -*- coding: utf-8 -*-
# license removed for brevity

import rospy
import serial
from inmoov_lxd_headcontrol.msg import head_control

#code结构体定义某控制量的当前值，定义其可以运动的范围，和每次控制后所能改变的控制量大小
class code():
    def __init__(self,code,min_area,max_area,ratio):
        self.code = code
        self.min = min_area
        self.max = max_area
        self.ratio = ratio
#头部控制量相关类
class headcode():
    
    #默认的初始化三个控制量 对应头部的左右运动 上下运动 与眼球的摆动
    def __init__(self,head_leftright,head_updown,head_eye):
        self.code_one = code(head_leftright,-125,125,25)
        self.code_two = code(head_updown,-125,125,50)
        self.code_three = code(head_eye,-125,125,50)
        self.finish = False
    #生成下一次的控制量
    def nextcode(self):
        if self.code_one.code == self.code_one.max:
            self.code_one.code = self.code_one.min
            if self.code_two.code == self.code_two.max:
                self.code_two.code = self.code_two.min
                if self.code_three.code == self.code_three.max:
                    self.finish = True
                else:
                    self.code_three.code = self.code_three.code + self.code_three.ratio
                    if self.code_three.code > self.code_three.max:
                        self.code_three.code = self.code_three.max
            else:
                self.code_two.code = self.code_two.code +self.code_two.ratio
                if self.code_two.code > self.code_two.max:
                    self.code_two.code = self.code_two.max
        else:
            self.code_one.code  = self.code_one.code + self.code_one.ratio
            if self.code_one.code > self.code_one.max:
                self.code_one.code = self.code_one.max
#串口类
class piserial():
    #打开串口
    def __init__(self):
        rospy.loginfo("ROS Serial init") #初始化串口节点
        self.port_name = rospy.get_param('~port','/dev/ttyUSB0')
        self.baud = int(rospy.get_param('~baud','115200'))
        self.timout=0.1
        self.serial=serial.Serial(self.port_name,self.baud,timeout=self.timout) #创建串口句柄。
        self.serial_good = True
        if self.serial.isOpen() == False:
            try:
                self.serial.open()
                rospy.loginfo('open serial successful.')
                rospy.sleep(0.2)

            except Exception, e:
                try:
                    self.port_name=rospy.get_param('~port','/dev/ttyUSB1')
                    self.serial=serial.Serial(self.port_name,self.baud,timeout=self.timout) #创建串口句柄。
                    self.serial.open()
                    rospy.loginfo('open serial successful.')
                    rospy.sleep(0.2)
                except Exception as e:
                    rospy.loginfo('open serial failed.')
                    #尝试使用2种地址名来打开串口 均失败时产生错误标识
                    self.serial_good = False
        else:
            pass
    #发送一个字节的函数
    def piserial_sendbyte(self,data):
        self.serial.write(chr(data))
    #发送多个字节的函数
    def piserial_sendmultibyte(self,data_list):
        for data in data_list:
            self.piserial_sendbyte(data)
	    #rospy.sleep(0.001)
        rospy.sleep(0.1)

def piserial_sender_node():
    pub = rospy.Publisher('piserial_send_data', head_control, queue_size=10)
    rospy.init_node('piserial_sender', anonymous=False)
    rospy.loginfo('wait for system starting up')
    #rate = rospy.Rate(0.16) # 10hz
    control_code = headcode(-125,-125,-125)
    ser = piserial()
    head_control_msg = head_control()
    prt = ser.serial.readline()
    #prt = ser.serial.readline()
    rospy.sleep(4)
    rate = rospy.Rate(0.16) # 10hz
    while not rospy.is_shutdown():
        if control_code.finish == False:
            senddata_list = list()
	    senddata_list.append(2)
	    senddata_list.append(1)
	    senddata_list.append(0)
	    senddata_list.append(0)
            senddata_list.append(control_code.code_one.code+128)
            senddata_list.append(-control_code.code_two.code+128)
            senddata_list.append(0)
	    senddata_list.append(0)
	    senddata_list.append(0)
	    senddata_list.append(control_code.code_three.code+128)  #zhongjian  youshu  di 4 ge  --> ::: :<---
            ser.piserial_sendmultibyte(senddata_list)

            head_control_msg.head_leftright = control_code.code_one.code+128
            head_control_msg.head_updown = control_code.code_two.code+128
            head_control_msg.head_lefteye = control_code.code_three.code+128
            head_control_msg.head_righteye = 0
            head_control_msg.head_finish = 0
            rospy.loginfo('send message:')
            rospy.loginfo('send message:%d,%d,%d'%(control_code.code_one.code,control_code.code_two.code,control_code.code_three.code))
	    #prt = ser.serial.readline()
	    #rospy.loginfo(prt)

            control_code.nextcode()

        else:
            head_control_msg.head_leftright = control_code.code_one.code+128
            head_control_msg.head_updown = control_code.code_two.code+128
            head_control_msg.head_lefteye = control_code.code_three.code+128
            head_control_msg.head_righteye = 0
            head_control_msg.head_finish = 1
            rospy.loginfo('send finish')
        pub.publish(head_control_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        piserial_sender_node()
    except rospy.ROSInterruptException:
        pass

