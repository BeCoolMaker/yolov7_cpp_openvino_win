#include "DetYolov7.h"


int main() {
  DetYolov7 det;
  std::string model_dir = "data/yolov7/yolov7.onnx";
  std::string label_txt = "data/yolov7/label.txt";
  double cof_threshold = 0.25;
  double nms_area_threshold = 0.45;
  det.init(model_dir, label_txt, cof_threshold, nms_area_threshold);

  cv::Mat im = cv::imread("data/yolov7/zidane.jpg");
  auto result = det.detector(im);
}
