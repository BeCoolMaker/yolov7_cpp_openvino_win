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

    int inf_img_w;  // ����ģ�͵����ݿ��
    int inf_img_h;  // ����ģ�͵����ݸ߶�
    float scale;    // ���ű���
    int padd_w;     // ���ź����Ļ�ɫ�߿���
    int padd_h;     // ���ź����Ļ�ɫ�߿�߶�
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

