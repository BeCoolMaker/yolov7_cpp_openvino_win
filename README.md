# yolov7_cpp_openvino_win
Windows下超简单的YOLOV7加速demo。使用openvino加速cpu进行推理，支持YOLOV7 d6、e6、w6，简化初学者操作，开箱即食。

## 准备环境
- Windows10-x64
- Visual Studio 2022
- OpenVINO 2022.1.0.643
- OpenCV454

## Windows下快速使用
- 准备好yolov7 onnx模型、label文件、预测图像（label为txt文件，标签换行分隔）
- git clone https://github.com/BeCoolMaker/yolov7_cpp_openvino_win.git
- 安装好openvino，把安装路径下的runtime中include\bin\lib文件分别拷贝至项目文件夹下
- 准备好opencv > 3.0,同样把include\bin\lib拷贝至根目录下
- vs打开.sln文件，2022版本之前的可以去改sln文件，适配2019、2017等版本
- 运行
```
DetYolov7 det;
std::string model_dir = "data/yolov7/yolov7.onnx";  //模型路径
std::string label_txt = "data/yolov7/label.txt";    //标签路径
double cof_threshold = 0.2;                         //置信度
double nms_area_threshold = 0.45;                   //NMS置信度
det.init(model_dir, label_txt, cof_threshold, nms_area_threshold);

cv::Mat im = cv::imread("data/yolov7/im.jpg");
auto result = det.detector(im);
```
## 部署步骤
#### Step 1. 初始化
- 注意是否缺乏依赖项
#### Step 2. 读取模型
- 注意根目录下的**plugins.xml**是否存在，否则会报错
- 导入onnx时batch设置为1，暂未测试多batch方案
#### Step 3. 设置为cpu推理，获取模型输入输出
- GPU方案建议使用TensorRT,而非openvino
#### Step 4. 创建一个输入请求inference request
#### Step 5. 将Mat转换为openvino tensor
- letterbox 将图像缩放至模型所需尺寸，边界用灰色缝补
- openvino所需格式为归一化后的rgb图像
#### Step 6. 推理模型
#### Step 7. Process output
- w6的模型输出是4层{160, 80, 40, 20}，v7的模型输出是3层{80, 40, 20}
#### Step 7.1 解析模型输出
```
# yolov7.onnx
output_tensor get_shape{1, 3, 80, 80, 85}
output_tensor get_shape{1, 3, 40, 40, 85}
output_tensor get_shape{1, 3, 20, 20, 85}
```
- 推理后的数据是一个一维数组，原始形状如上，我们的目标就是解析它
- yolov7.pt是用的coco数据集其中有80类，加上xywh,conf就是85，数据格式为[x,y,w,h,边框的confidence,80类的置信度得分]
#### Step 8. NMS
#### Step 9. 可视化

## 参考项目
- https://github.com/OpenVINO-dev-contest/YOLOv7_OpenVINO_cpp-python
- https://github.com/fb029ed/yolov5_cpp_openvino
