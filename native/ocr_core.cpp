#include "ocr_core.h"
#include "ocr_geometry.h"

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ocr-daemon");


// The intra-op pool size for every session: OCR_THREADS wins, otherwise the
// physical core count (see ocr_geometry.h).
int session_threads() {
  if (const char *override = std::getenv("OCR_THREADS")) return std::atoi(override);
  return ocr_intra_op_threads();
}

class Session {
 public:
  explicit Session(const std::string &path, int threads) : options_(), session_(nullptr) {
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (threads > 0) options_.SetIntraOpNumThreads(threads);
    // The CPU memory arena stays off by default. It earned its keep when
    // recognition ran six crops per Run; with one Run per crop the per-call
    // allocations are small and the arena bought nothing measurable while pinning
    // ~1.15 GiB of cached activations for the corpus' fifty-odd canvas widths
    // (D2 in ASSUMPTIONS.md). OCR_MEM_ARENA=1 leaves it on so the per-run
    // allocation share can be measured against the same binary.
    if (!std::getenv("OCR_MEM_ARENA")) options_.DisableCpuMemArena();
    session_ = Ort::Session(env, path.c_str(), options_);
    Ort::AllocatorWithDefaultOptions allocator;
    input_ = session_.GetInputNameAllocated(0, allocator).get();
    output_ = session_.GetOutputNameAllocated(0, allocator).get();
  }

  // Hands back the runtime's own output tensor so callers decode it in place: a
  // Run then costs no copy of the 11.6 MB detector map or the 5.7 MB recognition
  // logits, which used to be memcpy'd into a fresh vector every time.
  Ort::Value run(const std::vector<float> &input, const std::vector<int64_t> &shape) {
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float *>(input.data()),
        input.size(), shape.data(), shape.size());
    const char *inputs[] = {input_.c_str()};
    const char *outputs[] = {output_.c_str()};
    auto values = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    return std::move(values[0]);
  }

  std::vector<std::string> characters() {
    Ort::AllocatorWithDefaultOptions allocator;
    auto metadata = session_.GetModelMetadata();
    auto text = metadata.LookupCustomMetadataMapAllocated("character", allocator);
    if (!text) throw std::runtime_error("recognition model has no character metadata");
    std::vector<std::string> result{""};
    std::istringstream lines(text.get());
    for (std::string line; std::getline(lines, line);) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      result.push_back(line);
    }
    result.push_back(" ");
    return result;
  }

 private:
  Ort::SessionOptions options_;
  Ort::Session session_;
  std::string input_, output_;
};

// The pixel math lives in ocr_geometry.c and is shared with the C core, so both
// cores hand the recogniser the same crops.
//
// Fills a caller-owned NCHW canvas. Both buffers are kept between requests, so the
// 35 MB detector canvas and the scaled 8-bit copy are not re-faulted per request;
// only the right-hand pad columns are cleared, because ocr_nchw_normalize writes
// every other element (zero is the pad value RapidOCR uses).
void fill_nchw(std::vector<float> &out, std::vector<unsigned char> &scaled, const cv::Mat &bgr,
               int target_h, int target_w, int resized_w) {
  resized_w = std::clamp(resized_w, 1, target_w);
  const std::size_t needed = static_cast<std::size_t>(3) * target_h * target_w;
  if (out.size() != needed) out.resize(needed);
  const unsigned char *source = bgr.data;
  int cols = bgr.cols;
  if (resized_w != bgr.cols || target_h != bgr.rows) {
    scaled.resize(static_cast<std::size_t>(target_h) * resized_w * 3);
    ocr_resize_bilinear(bgr.data, bgr.rows, bgr.cols, scaled.data(), target_h, resized_w);
    source = scaled.data();
    cols = resized_w;
  }
  ocr_nchw_normalize(out.data(), source, cols, target_h, target_w);
  if (cols < target_w) {
    const std::size_t width = sizeof(float) * static_cast<std::size_t>(target_w - cols);
    for (int channel = 0; channel < 3; ++channel) {
      for (int y = 0; y < target_h; ++y) {
        std::memset(out.data() + ((static_cast<std::size_t>(channel) * target_h + y) * target_w + cols), 0, width);
      }
    }
  }
}

cv::Mat resize_within_bounds(const cv::Mat &image, int min_side, int max_side) {
  cv::Mat result = image;
  for (int pass = 0; pass < 2; ++pass) {
    int h = result.rows, w = result.cols;
    double ratio = 1.0;
    if (pass == 0 && std::max(h, w) > max_side) ratio = static_cast<double>(max_side) / std::max(h, w);
    if (pass == 1 && std::min(h, w) < min_side) ratio = static_cast<double>(min_side) / std::min(h, w);
    if (ratio == 1.0) continue;
    int rh = ocr_resize_side(h * ratio), rw = ocr_resize_side(w * ratio);
    if (rh <= 0 || rw <= 0) continue;
    cv::Mat resized(rh, rw, CV_8UC3);
    ocr_resize_bilinear(result.data, result.rows, result.cols, resized.data, rh, rw);
    result = std::move(resized);
  }
  return result;
}

cv::Mat vertical_padding(const cv::Mat &image) {
  int h = image.rows, w = image.cols;
  if (h > 30 && w / static_cast<double>(h) <= 8.0) return image;
  int new_h = std::max(static_cast<int>(w / 8.0), 30) * 2;
  int pad = static_cast<int>(std::abs(new_h - h) / 2.0);
  cv::Mat padded;
  cv::copyMakeBorder(image, padded, pad, pad, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return padded;
}

cv::Mat crop(const cv::Mat &image, const OcrBox &box) {
  // Doubles throughout, like the C core: a float rounding here flips the crop
  // size by a pixel, which is enough to change what the recogniser reads.
  const int width = static_cast<int>(std::max(ocr_distance(box.c[0], box.c[1]), ocr_distance(box.c[2], box.c[3])));
  const int height = static_cast<int>(std::max(ocr_distance(box.c[0], box.c[3]), ocr_distance(box.c[1], box.c[2])));
  if (width < 1 || height < 1) return {};
  const OcrPoint target[4] = {{0, 0}, {static_cast<double>(width), 0},
      {static_cast<double>(width), static_cast<double>(height)}, {0, static_cast<double>(height)}};
  double matrix[9];
  if (ocr_perspective_transform(target, box.c, matrix) != 0) return {};
  cv::Mat result(height, width, CV_8UC3);
  ocr_warp_cubic(image.data, image.rows, image.cols, result.data, height, width, matrix);
  if (result.rows * 1.0 / result.cols >= 1.5) cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
  return result;
}

class Engine {
 public:
  explicit Engine(const std::string &models)
      : det_(models + "/PP-OCRv6_det_small.onnx", session_threads()),
        rec_(models + "/PP-OCRv6_rec_small.onnx", session_threads()),
        cls_(models + "/ch_ppocr_mobile_v2.0_cls_mobile.onnx", session_threads()),
        characters_(rec_.characters()) {}

  std::string run(const unsigned char *data, std::size_t size) {
    using clock = std::chrono::steady_clock;
    const auto started = clock::now();
    cv::Mat encoded(1, static_cast<int>(size), CV_8U, const_cast<unsigned char *>(data));
    cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("unsupported image");
    image = vertical_padding(resize_within_bounds(image, 30, 2000));
    if (!image.isContinuous()) image = image.clone();
    const auto decoded_at = clock::now();
    auto boxes = detect(image);
    const auto detected_at = clock::now();
    if (std::getenv("OCR_DEBUG")) {
      std::cerr << "image " << image.cols << "x" << image.rows << " boxes " << boxes.size() << '\n';
      for (const auto &box : boxes) {
        std::cerr << "box";
        for (const auto &p : box.c) std::cerr << ' ' << p.x << ',' << p.y;
        std::cerr << '\n';
      }
    }
    std::vector<cv::Mat> crops;
    std::vector<std::size_t> proposed;
    for (const auto &box : boxes) {
      cv::Mat line = crop(image, box);
      if (line.empty()) continue;
      if (proposes_rotation(line)) proposed.push_back(crops.size());
      crops.push_back(std::move(line));
    }
    const auto cropped_at = clock::now();
    auto decoded = recognize(crops);
    const auto recognized_at = clock::now();
    // The 180-degree classifier is confident far more often than it is right on
    // very wide crops, so its verdict is a proposal: recognise the rotated crop
    // and keep whichever reading the recogniser scores higher.
    for (std::size_t index : proposed) {
      cv::Mat turned;
      cv::rotate(crops[index], turned, cv::ROTATE_180);
      auto alternative = recognize({turned});
      if (alternative[0].second > decoded[index].second) decoded[index] = alternative[0];
    }
    std::vector<std::string> lines;
    for (const auto &line : decoded) {
      if (line.first.find_first_not_of(" \t\r\n") != std::string::npos && line.second >= .5f) lines.push_back(line.first);
    }
    if (std::getenv("OCR_DEBUG")) {
      // Stage split, so an optimisation is aimed at the stage that owns the time.
      auto ms = [](clock::time_point from, clock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
      };
      std::cerr << "time decode " << ms(started, decoded_at) << " det " << ms(decoded_at, detected_at)
                << " crops+cls " << ms(detected_at, cropped_at) << " rec " << ms(cropped_at, recognized_at)
                << " lines " << lines.size() << '\n';
    }
    std::ostringstream output;
    for (std::size_t i = 0; i < lines.size(); ++i) { if (i) output << '\n'; output << lines[i]; }
    return output.str();
  }

 private:
  std::vector<OcrBox> detect(const cv::Mat &image) {
    int h = image.rows, w = image.cols, rh = 0, rw = 0;
    ocr_detection_canvas(h, w, &rh, &rw);
    if (std::getenv("OCR_DEBUG")) {
      std::cerr << "canvas " << rw << 'x' << rh << " " << (rw * rh / 1e6) << " Mpx\n";
    }
    fill_nchw(det_input_, det_scaled_, image, rh, rw, rw);
    auto output = det_.run(det_input_, {1, 3, rh, rw});
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    const int ph = static_cast<int>(shape[2]), pw = static_cast<int>(shape[3]);
    const float *probability = output.GetTensorData<float>();
    unsigned char *mask = ocr_detection_mask(probability, ph, pw);
    if (!mask) throw std::runtime_error(ocr_geometry_error());
    OcrPixelGroups foreground{nullptr, 0}, holes{nullptr, 0};
    const bool grouped = ocr_group_pixels(mask, ph, pw, 255, 1, 0, &foreground) == 0 &&
        ocr_group_pixels(mask, ph, pw, 0, 0, 1, &holes) == 0;
    std::free(mask);
    if (!grouped) {
      ocr_groups_free(&foreground);
      ocr_groups_free(&holes);
      throw std::runtime_error(ocr_geometry_error());
    }
    std::vector<OcrBox> found;
    std::vector<OcrPoint> points;
    const std::size_t total = foreground.count + holes.count;
    for (std::size_t index = 0; index < total && found.size() < total; ++index) {
      const OcrPixels *group = index < foreground.count
          ? &foreground.items[index] : &holes.items[index - foreground.count];
      points.resize(group->count);
      for (std::size_t i = 0; i < group->count; ++i) {
        points[i] = {static_cast<double>(group->x[i]), static_cast<double>(group->y[i])};
      }
      OcrRect rect;
      if (ocr_min_area_rect(points.data(), points.size(), &rect) != 0) continue;
      if (std::min(rect.width, rect.height) < 3) continue;
      OcrPoint raw[4];
      ocr_rect_points(&rect, raw);
      if (std::getenv("OCR_DEBUG")) {
        std::cerr << "raw";
        for (const auto &p : raw) std::cerr << ' ' << p.x << ',' << p.y;
        std::cerr << " score " << ocr_box_score(probability, pw, ph, raw) << '\n';
      }
      if (ocr_box_score(probability, pw, ph, raw) < .5f) continue;
      // RapidOCR offsets a polygon by area * 1.6 / perimeter; for a rectangle that is
      // the same as growing both sides by that distance.
      const double growth = rect.width * rect.height * 1.6 / (2 * (rect.width + rect.height));
      rect.width += 2 * growth;
      rect.height += 2 * growth;
      if (std::min(rect.width, rect.height) < 5) continue;
      ocr_rect_points(&rect, raw);
      OcrBox box;
      ocr_order_box(raw, &box);
      for (int i = 0; i < 4; ++i) {
        const double x = ocr_py_round(box.c[i].x / pw * w), y = ocr_py_round(box.c[i].y / ph * h);
        box.c[i].x = std::clamp(x, 0.0, static_cast<double>(w - 1));
        box.c[i].y = std::clamp(y, 0.0, static_cast<double>(h - 1));
      }
      if (ocr_distance(box.c[0], box.c[1]) <= 3 || ocr_distance(box.c[0], box.c[3]) <= 3) continue;
      found.push_back(box);
    }
    ocr_groups_free(&foreground);
    ocr_groups_free(&holes);
    ocr_sort_boxes(found.data(), found.size());
    return found;
  }

  // True when the direction classifier is at least 90% sure this crop is upside
  // down. That is a proposal only; the recogniser arbitrates it.
  bool proposes_rotation(const cv::Mat &image) {
    int width = std::min(192, static_cast<int>(std::ceil(48.0 * image.cols / image.rows)));
    fill_nchw(cls_input_, cls_scaled_, image, 48, 192, width);
    auto output = cls_.run(cls_input_, {1, 3, 48, 192});
    const float *scores = output.GetTensorData<float>();
    const std::size_t count = output.GetTensorTypeAndShapeInfo().GetElementCount();
    bool rotate = count == 2 && scores[1] > .9f && scores[1] > scores[0];
    if (std::getenv("OCR_DEBUG") && count == 2) std::cerr << "cls " << scores[0] << ' ' << scores[1] << " rotate " << rotate << '\n';
    return rotate;
  }

  // One Run per crop. Batching crops onto a canvas padded to the widest aspect in
  // the group made the recogniser 1.7x slower per crop -- the batched activations
  // no longer fit in cache -- and the padding could invent a whole phantom line,
  // so every crop is now read on a canvas of exactly its own aspect.
  std::vector<std::pair<std::string, float>> recognize(const std::vector<cv::Mat> &images) {
    std::vector<std::pair<std::string, float>> results(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
      const auto &image = images[i];
      const std::size_t width = static_cast<std::size_t>(48 * std::max(320.0 / 48, image.cols * 1.0 / image.rows));
      fill_nchw(rec_input_, rec_scaled_, image, 48, static_cast<int>(width), std::min(static_cast<int>(width), static_cast<int>(std::ceil(48.0 * image.cols / image.rows))));
      auto output = rec_.run(rec_input_, {1, 3, 48, static_cast<int64_t>(width)});
      const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
      int steps = static_cast<int>(shape[1]), classes = static_cast<int>(shape[2]);
      const float *data = output.GetTensorData<float>();
      int previous = -1; float score_sum = 0; int score_count = 0; std::string text;
      for (int step = 0; step < steps; ++step) {
        const float *row = data + static_cast<std::size_t>(step) * classes;
        int index = static_cast<int>(std::max_element(row, row + classes) - row);
        float score = row[index];
        if (index && index != previous && index < static_cast<int>(characters_.size())) { text += characters_[index]; score_sum += score; ++score_count; }
        previous = index;
      }
      results[i] = {text, score_count ? score_sum / score_count : 0};
    }
    return results;
  }

  // Kept between requests: the detector canvas alone is 35 MB of floats, and
  // allocating it fresh cost 28.4 ms per request against 7.4 ms reused, almost all
  // of it page faults (ASSUMPTIONS.md E).
  Session det_, rec_, cls_;
  std::vector<std::string> characters_;
  std::vector<float> det_input_, rec_input_, cls_input_;
  std::vector<unsigned char> det_scaled_, rec_scaled_, cls_scaled_;
};

char *copy_string(const std::string &value) {
  char *result = static_cast<char *>(std::malloc(value.size() + 1));
  if (result) std::memcpy(result, value.c_str(), value.size() + 1);
  return result;
}
}  // namespace

extern "C" void *ocr_create(const char *model_dir, char **error) {
  try { return new Engine(model_dir); } catch (const std::exception &e) { *error = copy_string(e.what()); return nullptr; }
}
extern "C" char *ocr_run(void *engine, const unsigned char *data, std::size_t size, char **error) {
  try { return copy_string(static_cast<Engine *>(engine)->run(data, size)); } catch (const std::exception &e) { *error = copy_string(e.what()); return nullptr; }
}
extern "C" void ocr_destroy(void *engine) { delete static_cast<Engine *>(engine); }
extern "C" void ocr_free(void *value) { std::free(value); }
