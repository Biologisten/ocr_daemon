#include "ocr_core.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  const char *models = std::getenv("OCR_MODEL_DIR");
  if (!models)
    return 2;
  char *error = nullptr;
  void *engine = ocr_create(models, &error);
  if (!engine) {
    std::cerr << error << '\n';
    ocr_free(error);
    return 1;
  }
  cv::Mat white(64, 256, CV_8UC3, cv::Scalar(255, 255, 255));
  std::vector<unsigned char> png;
  cv::imencode(".png", white, png);
  char *text = ocr_run(engine, png.data(), png.size(), &error);
  bool ok = text && std::strlen(text) == 0;
  if (!ok && error)
    std::cerr << error << '\n';
  ocr_free(text);
  ocr_free(error);
  ocr_destroy(engine);
  return ok ? 0 : 1;
}
