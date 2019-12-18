# 说明文件
- ReadMark.py  
用于读取标注点文件，给出任何一张图片的标注点信息数据
- ImageProcess.py  
用于处理带有1-2标注点的图片，得到点阵像素坐标与真实尺度映射数据
待做 hough变化那段的图像输出
- ImageData.py  
用于解读ImageProcess.py得到的点阵像素坐标与真实尺度映射数据
需要改下文件结构
- DataPreprocess.py  
用于根据所有图片的处理结果，整合处理为易于训练或拟合的训练集和测试集
- Fit.py  
包含所用到的拟合功能块