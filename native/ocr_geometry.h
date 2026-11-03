// Crop geometry shared by both OCR cores: the image operations the C++ core
// would otherwise take from OpenCV (resize, min-area rect, perspective warp,
// mask building) and the box ordering/scoring rules. Living in one C11 module
// means the C and C++ cores feed the recogniser identical crops, so a difference
// between their outputs can only come from the models or the frontend.
#ifndef OCR_GEOMETRY_H
#define OCR_GEOMETRY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// A packed 8-bit BGR buffer. The shared functions only borrow it.
typedef struct {
  int rows;
  int cols;
  unsigned char *data;  // rows * cols * 3, BGR
} OcrImage;

typedef struct {
  double x, y;
} OcrPoint;

typedef struct {
  OcrPoint c[4];
} OcrBox;

typedef struct {
  double cx, cy, ux, uy, vx, vy, width, height;
} OcrRect;

typedef struct {
  int *x;
  int *y;
  size_t count;
} OcrPixels;

typedef struct {
  OcrPixels *items;
  size_t count;
} OcrPixelGroups;

// Message for the last failure returned by this module.
const char *ocr_geometry_error(void);

// Python's round(): half to even. Detection coordinates must match RapidOCR.
int ocr_py_round(double value);
int ocr_resize_side(double value);
double ocr_distance(OcrPoint a, OcrPoint b);

// Intra-op thread count the cores should ask ONNX Runtime for, or 0 to leave
// the runtime's own default in place. ONNX Runtime sizes its pool from the
// logical core count, which on an SMT machine puts two pool threads on every
// physical core; measured on the benchmark host, half the logical count (the
// physical cores) is ~18% faster on every corpus image. Machines without SMT
// return 0, so their behaviour is unchanged.
int ocr_intra_op_threads(void);

// Detector canvas for an image of this size. RapidOCR scales the short side up
// to 736 px; the result is then capped to a pixel budget, so a thin screenshot
// cannot buy 4 Mpx of detector work for text the model reads just as well at a
// smaller scale. Both sides stay multiples of 32, like the rest of the pipeline,
// and the cap is a hard bound (scaling floors rather than rounds).
void ocr_detection_canvas(int height, int width, int *rows, int *cols);

// cv::resize(INTER_LINEAR) on 8-bit BGR: 11-bit fixed-point coefficients, a
// horizontal pass into int, then a vertical pass with rounding.
void ocr_resize_bilinear(const unsigned char *src, int src_rows, int src_cols,
                         unsigned char *dst, int dst_rows, int dst_cols);

// cv::warpPerspective(INTER_CUBIC, BORDER_REPLICATE) for an integer target rect.
// `matrix` maps destination pixels to source pixels.
void ocr_warp_cubic(const unsigned char *src, int src_rows, int src_cols,
                    unsigned char *dst, int rows, int cols, const double matrix[9]);

// Places a BGR image in an NCHW float canvas (target_h x target_w), scaling to
// [0, 1] - 0.5 / 0.5 and leaving the remainder of each row zero (the pad value).
void ocr_nchw_normalize(float *out, const unsigned char *bgr, int cols,
                        int target_h, int target_w);

// cv::threshold(0.3) followed by cv::dilate with a 2x2 kernel, anchor (0,0).
// Returns a malloc'd rows * cols mask, or NULL on allocation failure.
unsigned char *ocr_detection_mask(const float *probability, int rows, int cols);

// Labels `mask == value` with the given connectivity and collects each group's
// boundary pixels. findContours(RETR_LIST) reports both foreground components
// and holes, so callers pass value 0 with 4-connectivity for the holes.
int ocr_group_pixels(const unsigned char *mask, int rows, int cols, unsigned char value,
                     int eight_connected, int skip_border, OcrPixelGroups *groups);
void ocr_groups_free(OcrPixelGroups *groups);

// RapidOCR's order_points: sort by x, split into left/right pairs, sort each by y.
void ocr_order_box(const OcrPoint raw[4], OcrBox *box);

// Rotating calipers over the convex hull, the same construction as
// cv::minAreaRect. Sorts `points` in place.
int ocr_min_area_rect(OcrPoint *points, size_t count, OcrRect *rect);
void ocr_rect_points(const OcrRect *rect, OcrPoint out[4]);

// cv::fillConvexPoly + cv::mean: mean probability inside the box.
float ocr_box_score(const float *probability, int width, int height, const OcrPoint box[4]);

// cv::getPerspectiveTransform for the 4-point case, solved as a linear system.
int ocr_perspective_transform(const OcrPoint src[4], const OcrPoint dst[4], double matrix[9]);

// stable_sort by top-left y, then group rows (>= 10 px apart start a new line)
// and stable_sort by (line, x), matching RapidOCR's reading order.
void ocr_sort_boxes(OcrBox *boxes, size_t count);

#ifdef __cplusplus
}
#endif

#endif
