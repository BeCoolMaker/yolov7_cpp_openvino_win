#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include <ie/inference_engine.hpp>

struct DetRect {
  cv::Rect_<float> rect;
  std::string label;
  float prob;
};

class DetYolov7 {
   public:
    void init(std::string model_path, std::string label_path,
              double cof_threshold, double nms_area_threshold);
    std::vector<DetRect> detector(cv::Mat input);

    int inf_img_w;  // 推理模型的数据宽度
    int inf_img_h;  // 推理模型的数据高度
    float scale;    // 缩放比例
    int padd_w;     // 缩放后填充的灰色边框宽度
    int padd_h;     // 缩放后填充的灰色边框高度
    std::string _input_name;
    std::vector<std::string> labels;
    ov::Shape output_shape;
    double cof_thresh;
    double nms_thresh;

    ov::Output<const ov::Node> input_port;
    ov::CompiledModel compiled_model;

   private:
    std::vector<std::string> parse_label(std::string path);
    double sigmoid(double x);
    std::vector<int> get_anchors(int net_grid);
    void letterbox(cv::Mat& src, cv::Mat& dst);
    void xywh2xyxy(int x, int y, int w, int h, cv::Rect& rect);
    void parse_yolov7(const float* output_blob, int net_grid,
                      float cof_threshold, std::vector<cv::Rect>& o_Rect,
                      std::vector<float>& o_Rect_cof,
                      std::vector<std::string>& origin_label);
};

