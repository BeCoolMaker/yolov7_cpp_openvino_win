#include "DetYolov7.h"

double DetYolov7::sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

std::vector<std::string> DetYolov7::parse_label(std::string path) {
    std::vector<std::string> labels;
    std::ifstream in(path);
    std::string line;
    if (in) {
        while (std::getline(in, line)) {
            if (line[0] == '#')
                continue;
            labels.push_back(line);
        }
    } else {
        std::cout << "no such file:" << path << std::endl;
    }

    return labels;
}

std::vector<int> DetYolov7::get_anchors(int net_grid) {
    std::vector<int> a160, a80, a40, a20;
    auto out_info = compiled_model.outputs();
    // d6,e6,e6e using this
    if (out_info.size() == 3) {
        a80 = {12, 16, 19, 36, 40, 28};
        a40 = {36, 75, 76, 55, 72, 146};
        a20 = {142, 110, 192, 243, 459, 401};
    } else {
        a160 = {19, 27, 44, 40, 38, 94};
        a80 = {96, 68, 86, 152, 180, 137};
        a40 = {140, 301, 303, 264, 238, 542};
        a20 = {436, 615, 739, 380, 925, 792};
    }

    if (net_grid == 160) {
        return a160;
    } else if (net_grid == 80) {
        return a80;
    } else if (net_grid == 40) {
        return a40;
    } else if (net_grid == 20) {
        return a20;
    } else {
        throw std::exception("Net grid is error.");
    }
}

void DetYolov7::letterbox(cv::Mat& src, cv::Mat& dst) {
    // 将图像resize为模型输入尺寸，缩放比例为长宽中最小的，缩放完用灰色边框填充
    int in_w = src.cols;
    int in_h = src.rows;
    // 哪个缩放比例小选用哪个
    scale = std::min(float(inf_img_h) / in_h, float(inf_img_w) / in_w);
    int inside_w = round(in_w * scale);
    int inside_h = round(in_h * scale);
    // 内层图像resize
    cv::Mat resize_img;
    resize(src, resize_img, cv::Size(inside_w, inside_h));
    // 外层边框填充灰色
    padd_w = (inf_img_w - inside_w) / 2;
    padd_h = (inf_img_h - inside_h) / 2;

    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    cv::copyMakeBorder(resize_img, dst, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

void DetYolov7::xywh2xyxy(int x, int y, int w, int h, cv::Rect& rect) {
    // 将xywh恢复为缩放前比例
    x = (x - padd_w) / scale;
    y = (y - padd_h) / scale;
    w = w / scale;
    h = h / scale;
    // 将xywh转换为xyxy
    double r_x = x - w / 2;
    double r_y = y - h / 2;
    rect = cv::Rect(round(r_x), round(r_y), round(w), round(h));
}

void DetYolov7::init(std::string model_path, std::string label_path,
                     double cof_threshold, double nms_area_threshold) {
    cof_thresh = cof_threshold;
    nms_thresh = nms_area_threshold;

    labels = parse_label(label_path);
    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Read a model --------
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    // -------- Step 3. Loading a model to the device --------
    compiled_model = core.compile_model(model, "CPU");

    // Get input port for model with one input
    input_port = compiled_model.input();
    auto input_shape = model->input().get_shape();
    OPENVINO_ASSERT(model->inputs().size() == 1, "模型输入的batch只能为1");
    OPENVINO_ASSERT(input_shape.size() == 4,
                    "模型输入的维度为NCWH，该模型维度不为4");
    
    inf_img_h = input_shape[2];
    inf_img_w = input_shape[3];
}

std::vector<DetRect> DetYolov7::detector(cv::Mat src) {
    OPENVINO_ASSERT(!src.empty(), "Image is empty");
    if (src.channels() == 1) {
        cvtColor(src, src, cv::COLOR_GRAY2BGR);
    }
    cv::Mat input, img; 
    cvtColor(src, input, cv::COLOR_BGR2RGB);
    // -------- Step 4. Create an infer request --------
    auto infer_request = compiled_model.create_infer_request();

    // -------- Step 5. Mat convert to openvino tensor --------
    letterbox(input, img);
    float* input_data = new float[inf_img_h * inf_img_w * 3];
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < inf_img_h; h++)
            for (int w = 0; w < inf_img_w; w++) {
              int out_index = c * inf_img_h * inf_img_w + h * inf_img_w + w;
              input_data[out_index] =
                  float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
    ov::Tensor input_tensor(input_port.get_element_type(),
                            input_port.get_shape(), input_data);
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 6.Inference model --------
    infer_request.infer();
    delete[] input_data;

    // -------- Step 7. Process output --------
    std::vector<cv::Rect> origin_rect;
    std::vector<float> origin_rect_cof;
    std::vector<std::string> origin_label;
    std::vector<int> net_grids;

    auto out_info = compiled_model.outputs();

    // Check whether it is v7-e6 or v7 model
    if (out_info.size() == 3) {
        net_grids = {80, 40, 20};
    } else {
        net_grids = {160, 80, 40, 20};
    }

    for (size_t i = 0; i < net_grids.size(); i++) {
        auto output_tensor = infer_request.get_output_tensor(i);
        std::cout << "output_tensor get_shape" << output_tensor.get_shape() << std::endl;
        const float* _result = output_tensor.data<const float>();
        parse_yolov7(_result, net_grids[i], cof_thresh, origin_rect,
                     origin_rect_cof, origin_label);
    }

    // -------- Step 8. NMS --------
    std::vector<int> final_id;
    cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, cof_thresh, nms_thresh,
                      final_id);

    // -------- Step 9. Visual output --------
    std::vector<DetRect> det_rects;
    for (int i = 0; i < final_id.size(); ++i) {
        int index = final_id[i];
        DetRect det;
        det.rect = origin_rect[index];
        det.prob = origin_rect_cof[index];
        det.label = origin_label[index];
        det_rects.push_back(det);
    }

    for (size_t i = 0; i < det_rects.size(); i++) {
        auto det = det_rects[i];
        int x0 = det.rect.x;
        int y0 = det.rect.y;
        int x1 = x0 + det.rect.width;
        int y1 = y0 + det.rect.height;
        cv::rectangle(src, cv::Point(x0, y0), cv::Point(x1, y1),
                      cv::Scalar(0, 255, 0), 1);
        std::string prob = cv::format("%.2f", det.prob);
        std::string label = det.label + ": " + prob;

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.25, 1, &baseLine);
        cv::rectangle(src, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
                      cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(src, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(), 1.5);
    }

    cv::imwrite("pred.jpg", src);
    return det_rects;
}

void DetYolov7::parse_yolov7(const float* output_blob, int net_grid,
                             float cof_threshold, std::vector<cv::Rect>& o_Rect,
                             std::vector<float>& o_Rect_cof,
                             std::vector<std::string>& origin_label) {
    std::vector<int> anchors = get_anchors(net_grid);
    int item_size = 5 + labels.size();
    int anchor_n = 3;
    for (int n = 0; n < anchor_n; ++n)
        for (int i = 0; i < net_grid; ++i)
            for (int j = 0; j < net_grid; ++j) {
              int buf_pos = n * net_grid * net_grid * item_size +
                            i * net_grid * item_size + j * item_size;
              double box_prob = output_blob[buf_pos + 4];
              box_prob = sigmoid(box_prob);
              // 框置信度不满足则整体置信度不满足
              if (box_prob < cof_threshold) continue;

              // 获取得分最高的类别idx
              double max_prob = 0;
              int idx = 0;
              for (int t = 5; t < item_size; ++t) {
                double tp = output_blob[buf_pos + t];
                tp = sigmoid(tp);
                if (tp > max_prob) {
                  max_prob = tp;
                  idx = t;
                }
              }
              float cof = box_prob * max_prob;
              // 对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
              if (cof < cof_threshold) continue;

              double x = output_blob[buf_pos + 0];
              double y = output_blob[buf_pos + 1];
              double w = output_blob[buf_pos + 2];
              double h = output_blob[buf_pos + 3];

              x = (sigmoid(x) * 2 - 0.5 + j) * float(inf_img_w) / net_grid;
              y = (sigmoid(y) * 2 - 0.5 + i) * float(inf_img_h) / net_grid;
              w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
              h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

              cv::Rect rect;
              xywh2xyxy(x, y, w, h, rect);

              o_Rect.push_back(rect);
              o_Rect_cof.push_back(cof);
              origin_label.push_back(labels[idx - 5]);
            }
}
