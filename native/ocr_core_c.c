// Independent OCR core in C11: ONNX Runtime's C API plus the image processing
// that the C++ core gets from OpenCV (OpenCV 5 no longer exposes a C API). It
// implements the same C ABI as ocr_core.cpp, so native/c_daemon.c links against
// this core instead of the C++ one.
#define _POSIX_C_SOURCE 200809L

#include "ocr_core.h"
#include "ocr_geometry.h"

#include <onnxruntime/onnxruntime_c_api.h>

#include <stdio.h>

#include <jpeglib.h>
#include <png.h>

#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ERROR_SIZE 1024
#define COEF_BITS 11
#define COEF_SCALE (1 << COEF_BITS)

static char g_error[ERROR_SIZE];
static const OrtApi *g_ort;
static OrtEnv *g_env;

static int fail(const char *message) {
  snprintf(g_error, sizeof g_error, "%s", message);
  return -1;
}

static int fail_ort(const char *what, OrtStatus *status) {
  const char *message = status ? g_ort->GetErrorMessage(status) : "unknown error";
  snprintf(g_error, sizeof g_error, "%s: %s", what, message);
  if (status) g_ort->ReleaseStatus(status);
  return -1;
}

static void allocator_free(OrtAllocator *allocator, void *value) {
  OrtStatus *status = g_ort->AllocatorFree(allocator, value);
  if (status) g_ort->ReleaseStatus(status);
}

// ---------------------------------------------------------------- image (BGR)

// The shared geometry module owns the pixel math; these are the buffers it
// writes into and the ownership helpers around them.
typedef OcrImage Image;
typedef OcrPoint Point;
typedef OcrBox Box;
typedef OcrRect Rect;
typedef OcrPixels Pixels;
typedef OcrPixelGroups PixelGroups;

static void image_free(Image *image) {
  free(image->data);
  image->data = NULL;
  image->rows = image->cols = 0;
}

static int image_create(Image *image, int rows, int cols) {
  image->rows = rows;
  image->cols = cols;
  image->data = malloc((size_t)rows * (size_t)cols * 3);
  if (!image->data) return fail("out of memory");
  return 0;
}

static int image_copy(const Image *src, Image *dst) {
  if (image_create(dst, src->rows, src->cols) != 0) return -1;
  memcpy(dst->data, src->data, (size_t)src->rows * src->cols * 3);
  return 0;
}

static int image_resize(const Image *src, Image *dst, int dst_rows, int dst_cols) {
  if (dst_rows <= 0 || dst_cols <= 0) return fail("invalid resize size");
  if (image_create(dst, dst_rows, dst_cols) != 0) return -1;
  ocr_resize_bilinear(src->data, src->rows, src->cols, dst->data, dst_rows, dst_cols);
  return 0;
}

// cv::copyMakeBorder with BORDER_CONSTANT (black) on top and bottom.
static int image_pad_vertical(const Image *src, Image *dst, int pad) {
  if (image_create(dst, src->rows + 2 * pad, src->cols) != 0) return -1;
  memset(dst->data, 0, (size_t)dst->rows * dst->cols * 3);
  memcpy(dst->data + (size_t)pad * dst->cols * 3, src->data, (size_t)src->rows * src->cols * 3);
  return 0;
}

static int image_rotate_90_ccw(const Image *src, Image *dst) {
  if (image_create(dst, src->cols, src->rows) != 0) return -1;
  for (int y = 0; y < src->rows; ++y) {
    for (int x = 0; x < src->cols; ++x) {
      unsigned char *out = dst->data + ((size_t)(src->cols - 1 - x) * dst->cols + y) * 3;
      memcpy(out, src->data + ((size_t)y * src->cols + x) * 3, 3);
    }
  }
  return 0;
}

// -------------------------------------------------------------------- decoding

typedef struct {
  struct jpeg_error_mgr base;
  jmp_buf jump;
} JpegError;

static void jpeg_bail(j_common_ptr info) { longjmp(((JpegError *)info->err)->jump, 1); }

static int decode_png(const unsigned char *data, size_t size, Image *image) {
  png_image png;
  memset(&png, 0, sizeof png);
  png.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&png, data, size)) return fail("unsupported image");
  // libpng writes BGR straight into the image: the bytes are already in the order
  // the pipeline wants, so there is no second full-size decode buffer and no
  // per-pixel channel swap (a screenshot is ~2 Mpx of that per request).
  png.format = PNG_FORMAT_BGR;
  if (image_create(image, (int)png.height, (int)png.width) != 0) {
    png_image_free(&png);
    return -1;
  }
  if (!png_image_finish_read(&png, NULL, image->data, 0, NULL)) {
    image_free(image);
    png_image_free(&png);
    return fail("unsupported image");
  }
  png_image_free(&png);
  return 0;
}

static int decode_jpeg(const unsigned char *data, size_t size, Image *image) {
  struct jpeg_decompress_struct info;
  JpegError error;
  unsigned char *rgb = NULL;
  image->data = NULL;
  image->rows = image->cols = 0;
  info.err = jpeg_std_error(&error.base);
  error.base.error_exit = jpeg_bail;
  if (setjmp(error.jump)) {
    jpeg_destroy_decompress(&info);
    free(rgb);
    image_free(image);
    return fail("unsupported image");
  }
  jpeg_create_decompress(&info);
  jpeg_mem_src(&info, data, (unsigned long)size);
  jpeg_read_header(&info, TRUE);
  // libjpeg-turbo decodes the usual YCbCr and grayscale sources straight to BGR,
  // which removes both the second full-size buffer and the channel-swap pass.
  // CMYK/YCCK keep the old RGB path, since libjpeg has no BGR conversion there.
  const int direct = info.jpeg_color_space != JCS_CMYK && info.jpeg_color_space != JCS_YCCK;
  info.out_color_space = direct ? JCS_EXT_BGR : JCS_RGB;
  jpeg_start_decompress(&info);
  const int rows = (int)info.output_height, cols = (int)info.output_width;
  if (direct) {
    if (image_create(image, rows, cols) != 0) {
      jpeg_destroy_decompress(&info);
      return -1;
    }
    while (info.output_scanline < info.output_height) {
      JSAMPROW scanlines[1] = {image->data + (size_t)info.output_scanline * cols * 3};
      jpeg_read_scanlines(&info, scanlines, 1);
    }
  } else {
    rgb = malloc((size_t)rows * (size_t)cols * 3);
    if (!rgb) {
      jpeg_destroy_decompress(&info);
      return fail("out of memory");
    }
    while (info.output_scanline < info.output_height) {
      JSAMPROW scanlines[1] = {rgb + (size_t)info.output_scanline * cols * 3};
      jpeg_read_scanlines(&info, scanlines, 1);
    }
    if (image_create(image, rows, cols) == 0) {
      for (size_t i = 0; i < (size_t)rows * (size_t)cols; ++i) {  // RGB -> BGR
        image->data[i * 3] = rgb[i * 3 + 2];
        image->data[i * 3 + 1] = rgb[i * 3 + 1];
        image->data[i * 3 + 2] = rgb[i * 3];
      }
    }
    free(rgb);
    rgb = NULL;
  }
  jpeg_finish_decompress(&info);
  jpeg_destroy_decompress(&info);
  return 0;
}

// What cv::imdecode(IMREAD_COLOR) accepts here: PNG (alpha dropped, no
// compositing) and JPEG (decoded as BGR).
static int decode_image(const unsigned char *data, size_t size, Image *image) {
  static const unsigned char png_magic[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  if (size >= sizeof png_magic && memcmp(data, png_magic, sizeof png_magic) == 0) return decode_png(data, size, image);
  if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) return decode_jpeg(data, size, image);
  return fail("unsupported image");
}

// ---------------------------------------------------------- pipeline resizing

static int resize_within_bounds(const Image *src, int min_side, int max_side, Image *out) {
  Image current;
  if (image_copy(src, &current) != 0) return -1;
  for (int pass = 0; pass < 2; ++pass) {
    const int h = current.rows, w = current.cols;
    const int largest = h > w ? h : w, smallest = h < w ? h : w;
    double ratio = 1.0;
    if (pass == 0 && largest > max_side) ratio = (double)max_side / largest;
    if (pass == 1 && smallest < min_side) ratio = (double)min_side / smallest;
    if (ratio == 1.0) continue;
    const int rh = ocr_resize_side(h * ratio), rw = ocr_resize_side(w * ratio);
    if (rh <= 0 || rw <= 0) continue;
    Image resized;
    if (image_resize(&current, &resized, rh, rw) != 0) {
      image_free(&current);
      return -1;
    }
    image_free(&current);
    current = resized;
  }
  *out = current;
  return 0;
}

static int vertical_padding(const Image *src, Image *out) {
  const int h = src->rows, w = src->cols;
  if (h > 30 && (double)w / h <= 8.0) return image_copy(src, out);
  const int eighth = (int)(w / 8.0);
  const int target = (eighth > 30 ? eighth : 30) * 2;
  const int pad = (int)(fabs((double)target - h) / 2.0);
  return image_pad_vertical(src, out, pad);
}

// Places one image in an NCHW float canvas (target_h x target_w), resized to
// resized_w (negative means target_w) and zero padded on the right.
// Growing a canvas zero-fills only the new tail; fill_nchw clears just the pad
// columns, because ocr_nchw_normalize writes every other element. Keeping the
// buffers on the engine is what stops a request re-faulting 35 MB of floats and
// a 10 MB scaled copy (the detector resize) on every call.
static int ensure_floats(float **buffer, size_t *capacity, size_t elements) {
  if (*capacity >= elements) return 0;
  float *grown = realloc(*buffer, elements * sizeof(float));
  if (!grown) return fail("out of memory");
  memset(grown + *capacity, 0, (elements - *capacity) * sizeof(float));
  *buffer = grown;
  *capacity = elements;
  return 0;
}

static int ensure_bytes(unsigned char **buffer, size_t *capacity, size_t bytes) {
  if (*capacity >= bytes) return 0;
  unsigned char *grown = realloc(*buffer, bytes);
  if (!grown) return fail("out of memory");
  *buffer = grown;
  *capacity = bytes;
  return 0;
}

static int fill_nchw(float **buffer, size_t *capacity, unsigned char **scaled,
                     size_t *scaled_capacity, const Image *image,
                     int target_h, int target_w, int resized_w) {
  if (resized_w < 1) resized_w = 1;
  if (resized_w > target_w) resized_w = target_w;
  if (ensure_floats(buffer, capacity, (size_t)3 * target_h * target_w) != 0) return -1;
  const unsigned char *source = image->data;
  int cols = image->cols;
  if (resized_w != image->cols || target_h != image->rows) {
    if (ensure_bytes(scaled, scaled_capacity, (size_t)target_h * resized_w * 3) != 0) return -1;
    ocr_resize_bilinear(image->data, image->rows, image->cols, *scaled, target_h, resized_w);
    source = *scaled;
    cols = resized_w;
  }
  ocr_nchw_normalize(*buffer, source, cols, target_h, target_w);
  if (cols < target_w) {
    const size_t row = sizeof(float) * (size_t)(target_w - cols);
    for (int channel = 0; channel < 3; ++channel) {
      for (int y = 0; y < target_h; ++y) {
        memset(*buffer + ((size_t)channel * target_h + y) * target_w + cols, 0, row);
      }
    }
  }
  return 0;
}

// -------------------------------------------------------------- ONNX sessions

typedef struct {
  OrtSession *session;
  char *input_name;
  char *output_name;
} Session;

typedef struct {
  float *data;
  int64_t dims[8];
  int dim_count;
  OrtValue *value;  // set when `data` points into the runtime's buffer
} Tensor;

static void tensor_free(Tensor *tensor) {
  if (tensor->value) {
    g_ort->ReleaseValue(tensor->value);
    tensor->value = NULL;
  } else {
    free(tensor->data);
  }
  tensor->data = NULL;
  tensor->dim_count = 0;
}

static int session_create(Session *session, const char *path) {
  OrtSessionOptions *options = NULL;
  OrtAllocator *allocator = NULL;
  OrtStatus *status = g_ort->CreateSessionOptions(&options);
  memset(session, 0, sizeof *session);
  if (status) return fail_ort("cannot create session options", status);
  status = g_ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
  // OCR_THREADS overrides; otherwise the physical core count, because the
  // runtime's own default oversubscribes SMT siblings (see ocr_geometry.h).
  if (!status) {
    const char *override = getenv("OCR_THREADS");
    const int threads = override ? atoi(override) : ocr_intra_op_threads();
    if (threads > 0) status = g_ort->SetIntraOpNumThreads(options, threads);
  }
  // The CPU memory arena stays off by default: with one Run per crop it bought
  // nothing measurable and pins ~1.15 GiB for the corpus' many canvas widths
  // (D2). OCR_MEM_ARENA=1 leaves it on, so the per-run allocation share can be
  // measured against the same binary.
  if (!status && !getenv("OCR_MEM_ARENA")) status = g_ort->DisableCpuMemArena(options);
  if (!status) status = g_ort->CreateSession(g_env, path, options, &session->session);
  g_ort->ReleaseSessionOptions(options);
  if (status) return fail_ort("cannot open model", status);
  if (g_ort->GetAllocatorWithDefaultOptions(&allocator) != NULL) return fail("cannot get allocator");
  if (g_ort->SessionGetInputName(session->session, 0, allocator, &session->input_name) != NULL) return fail("cannot read input name");
  if (g_ort->SessionGetOutputName(session->session, 0, allocator, &session->output_name) != NULL) return fail("cannot read output name");
  return 0;
}

static void session_free(Session *session) {
  OrtAllocator *allocator = NULL;
  if (!session->session) return;
  if (g_ort->GetAllocatorWithDefaultOptions(&allocator) == NULL) {
    if (session->input_name) allocator_free(allocator, session->input_name);
    if (session->output_name) allocator_free(allocator, session->output_name);
  }
  g_ort->ReleaseSession(session->session);
  memset(session, 0, sizeof *session);
}

static int session_run(Session *session, const float *input, size_t elements, const int64_t *shape, int dim_count, Tensor *out) {
  OrtMemoryInfo *memory = NULL;
  OrtValue *input_value = NULL, *output_value = NULL;
  OrtTensorTypeAndShapeInfo *info = NULL;
  OrtStatus *status;
  size_t count = 0;
  memset(out, 0, sizeof *out);
  status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory);
  if (status) return fail_ort("cannot create memory info", status);
  status = g_ort->CreateTensorWithDataAsOrtValue(memory, (void *)input, elements * sizeof(float), shape,
      (size_t)dim_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_value);
  if (status) {
    g_ort->ReleaseMemoryInfo(memory);
    return fail_ort("cannot create input tensor", status);
  }
  const char *inputs[1] = {session->input_name};
  const char *outputs[1] = {session->output_name};
  status = g_ort->Run(session->session, NULL, inputs, (const OrtValue *const *)&input_value, 1, outputs, 1, &output_value);
  g_ort->ReleaseValue(input_value);
  g_ort->ReleaseMemoryInfo(memory);
  if (status) return fail_ort("inference failed", status);
  status = g_ort->GetTensorTypeAndShape(output_value, &info);
  if (status) {
    g_ort->ReleaseValue(output_value);
    return fail_ort("cannot read output shape", status);
  }
  status = g_ort->GetDimensionsCount(info, (size_t *)&count);
  if (!status) out->dim_count = (int)count;
  status = status ? status : g_ort->GetDimensions(info, out->dims, count);
  size_t element_count = 0;
  status = status ? status : g_ort->GetTensorShapeElementCount(info, &element_count);
  void *raw = NULL;
  status = status ? status : g_ort->GetTensorMutableData(output_value, &raw);
  g_ort->ReleaseTensorTypeAndShapeInfo(info);
  if (status) {
    g_ort->ReleaseValue(output_value);
    return fail_ort("cannot read output tensor", status);
  }
  // Borrow the runtime's buffer rather than copying it: the detector map alone is
  // 11.6 MB and the recognition logits 5.7 MB per crop. tensor_free releases it.
  (void)element_count;
  out->data = (float *)raw;
  out->value = output_value;
  return 0;
}

// --------------------------------------------------------------- geometry math

// Crop geometry lives in ocr_geometry.c and is shared with the C++ core, so a
// difference between the two cores can no longer come from their crops.
static int min_area_rect(Point *points, size_t count, Rect *rect) {
  return ocr_min_area_rect(points, count, rect) == 0 ? 0 : fail(ocr_geometry_error());
}

static int warp_cubic(const Image *src, Image *dst, const double matrix[9], int rows, int cols) {
  if (rows < 1 || cols < 1) return fail("invalid warp size");
  if (image_create(dst, rows, cols) != 0) return -1;
  ocr_warp_cubic(src->data, src->rows, src->cols, dst->data, rows, cols, matrix);
  return 0;
}

// ------------------------------------------------------------------ OCR engine

typedef struct {
  Session det, rec;
  char **characters;
  size_t character_count;
  float *det_input, *rec_input;
  size_t det_capacity, rec_capacity;
  // The scaled 8-bit copies the canvases are built from, also kept between
  // requests: the detector one is ~10 MB and was a fresh malloc per request.
  unsigned char *det_scaled, *rec_scaled;
  size_t det_scaled_capacity, rec_scaled_capacity;
} Engine;

static void engine_free(Engine *engine) {
  if (!engine) return;
  session_free(&engine->det);
  session_free(&engine->rec);
  for (size_t i = 0; i < engine->character_count; ++i) free(engine->characters[i]);
  free(engine->characters);
  free(engine->det_input);
  free(engine->rec_input);
  free(engine->det_scaled);
  free(engine->rec_scaled);
  free(engine);
}

static int load_characters(Engine *engine) {
  OrtAllocator *allocator = NULL;
  OrtModelMetadata *metadata = NULL;
  char *value = NULL;
  if (g_ort->GetAllocatorWithDefaultOptions(&allocator) != NULL) return fail("cannot get allocator");
  if (g_ort->SessionGetModelMetadata(engine->rec.session, &metadata) != NULL) return fail("cannot read model metadata");
  if (g_ort->ModelMetadataLookupCustomMetadataMap(metadata, allocator, "character", &value) != NULL ||
      !value) {
    if (metadata) g_ort->ReleaseModelMetadata(metadata);
    return fail("recognition model has no character metadata");
  }
  size_t capacity = 512, count = 0;
  char **characters = malloc(capacity * sizeof(char *));
  if (!characters) {
    allocator_free(allocator, value);
    g_ort->ReleaseModelMetadata(metadata);
    return fail("out of memory");
  }
  characters[count++] = strdup("");
  const char *line = value;
  while (*line) {
    const char *end = strchr(line, '\n');
    size_t length = end ? (size_t)(end - line) : strlen(line);
    if (length && line[length - 1] == '\r') --length;
    char *copy = malloc(length + 1);
    if (!copy) break;
    memcpy(copy, line, length);
    copy[length] = '\0';
    if (count + 1 >= capacity) {
      capacity *= 2;
      char **grown = realloc(characters, capacity * sizeof(char *));
      if (!grown) {
        free(copy);
        break;
      }
      characters = grown;
    }
    characters[count++] = copy;
    if (!end) break;
    line = end + 1;
  }
  if (count + 1 >= capacity) {
    char **grown = realloc(characters, (capacity + 1) * sizeof(char *));
    if (grown) characters = grown;
  }
  characters[count++] = strdup(" ");  // RapidOCR appends a space class
  engine->characters = characters;
  engine->character_count = count;
  allocator_free(allocator, value);
  g_ort->ReleaseModelMetadata(metadata);
  return 0;
}

static int detect(Engine *engine, const Image *image, Box **boxes, size_t *box_count) {
  const int h = image->rows, w = image->cols;
  int rh = 0, rw = 0;
  ocr_detection_canvas(h, w, &rh, &rw);
  if (fill_nchw(&engine->det_input, &engine->det_capacity, &engine->det_scaled,
                &engine->det_scaled_capacity, image, rh, rw, rw) != 0) return -1;
  const int64_t shape[4] = {1, 3, rh, rw};
  Tensor output;
  if (session_run(&engine->det, engine->det_input, (size_t)3 * rh * rw, shape, 4, &output) != 0) return -1;
  if (output.dim_count != 4) {
    tensor_free(&output);
    return fail("unexpected detector output shape");
  }
  const int ph = (int)output.dims[2], pw = (int)output.dims[3];
  unsigned char *dilated = ocr_detection_mask(output.data, ph, pw);
  if (!dilated) {
    tensor_free(&output);
    return fail(ocr_geometry_error());
  }

  PixelGroups foreground = {NULL, 0}, holes = {NULL, 0};
  int result = ocr_group_pixels(dilated, ph, pw, 255, 1, 0, &foreground);
  if (result == 0) result = ocr_group_pixels(dilated, ph, pw, 0, 0, 1, &holes);
  free(dilated);
  if (result != 0) {
    ocr_groups_free(&foreground);
    ocr_groups_free(&holes);
    tensor_free(&output);
    return -1;
  }

  const size_t total = foreground.count + holes.count;
  Box *found = malloc(sizeof(Box) * (total ? total : 1));
  if (!found) {
    ocr_groups_free(&foreground);
    ocr_groups_free(&holes);
    tensor_free(&output);
    return fail("out of memory");
  }
  size_t count = 0;
  for (size_t group_index = 0; group_index < total; ++group_index) {
    const PixelGroups *source = group_index < foreground.count ? &foreground : &holes;
    const Pixels *group = &source->items[group_index < foreground.count ? group_index : group_index - foreground.count];
    Point *points = malloc(sizeof(Point) * group->count);
    if (!points) break;
    for (size_t i = 0; i < group->count; ++i) {
      points[i].x = group->x[i];
      points[i].y = group->y[i];
    }
    Rect rect;
    if (min_area_rect(points, group->count, &rect) == 0) {
      free(points);
      if ((rect.width < rect.height ? rect.width : rect.height) < 3) continue;
      Point raw[4];
      ocr_rect_points(&rect, raw);
      if (ocr_box_score(output.data, pw, ph, raw) < .5f) continue;
      // RapidOCR grows the polygon by area * 1.6 / perimeter; for a rectangle
      // that is the same as growing both sides by that distance.
      const double growth = rect.width * rect.height * 1.6 / (2 * (rect.width + rect.height));
      rect.width += 2 * growth;
      rect.height += 2 * growth;
      if ((rect.width < rect.height ? rect.width : rect.height) < 5) continue;
      ocr_rect_points(&rect, raw);
      Box box;
      ocr_order_box(raw, &box);
      int ok = 1;
      for (int i = 0; i < 4; ++i) {
        double x = ocr_py_round(box.c[i].x / pw * w), y = ocr_py_round(box.c[i].y / ph * h);
        box.c[i].x = x < 0 ? 0 : x > w - 1 ? w - 1 : x;
        box.c[i].y = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
      }
      if (ocr_distance(box.c[0], box.c[1]) <= 3 || ocr_distance(box.c[0], box.c[3]) <= 3) ok = 0;
      if (ok && count < total) found[count++] = box;
      continue;
    }
    free(points);
  }
  ocr_groups_free(&foreground);
  ocr_groups_free(&holes);
  tensor_free(&output);
  ocr_sort_boxes(found, count);
  *boxes = found;
  *box_count = count;
  return 0;
}

static int crop_box(const Image *image, const Box *box, Image *out) {
  const int width = (int)fmax(ocr_distance(box->c[0], box->c[1]), ocr_distance(box->c[2], box->c[3]));
  const int height = (int)fmax(ocr_distance(box->c[0], box->c[3]), ocr_distance(box->c[1], box->c[2]));
  if (width < 1 || height < 1) return 1;  // empty crop, caller skips it
  Point target[4] = {{0, 0}, {width, 0}, {width, height}, {0, height}};
  double matrix[9];
  if (ocr_perspective_transform(target, box->c, matrix) != 0) return -1;
  Image warped;
  if (warp_cubic(image, &warped, matrix, height, width) != 0) return -1;
  if ((double)warped.rows / warped.cols >= 1.5) {
    Image rotated;
    if (image_rotate_90_ccw(&warped, &rotated) != 0) {
      image_free(&warped);
      return -1;
    }
    image_free(&warped);
    warped = rotated;
  }
  *out = warped;
  return 0;
}

typedef struct {
  char *text;
  float score;
} Line;

// One Run per crop (see the C++ core): batching crops onto a canvas padded to
// the widest aspect in the group cost 1.7x per crop and let the padding invent a
// whole phantom line, so every crop is read on a canvas of its own aspect.
static int recognize(Engine *engine, const Image *crops, size_t count, Line **lines) {
  Line *results = calloc(count ? count : 1, sizeof(Line));
  if (!results) return fail("out of memory");
  for (size_t i = 0; i < count; ++i) {
    const Image *image = &crops[i];
    const double target = 320.0 / 48 > (double)image->cols / image->rows ? 320.0 / 48 : (double)image->cols / image->rows;
    size_t width = (size_t)(48 * target);
    if (width < 1) width = 1;
    int resized_w = (int)ceil(48.0 * image->cols / image->rows);
    if (resized_w > (int)width) resized_w = (int)width;
    if (fill_nchw(&engine->rec_input, &engine->rec_capacity, &engine->rec_scaled,
                  &engine->rec_scaled_capacity, image, 48, (int)width, resized_w) != 0) break;
    const int64_t shape[4] = {1, 3, 48, (int64_t)width};
    Tensor output;
    if (session_run(&engine->rec, engine->rec_input, (size_t)3 * 48 * width, shape, 4, &output) != 0) break;
    const int steps = (int)output.dims[1], classes = (int)output.dims[2];
    int previous = -1, score_count = 0;
    float score_sum = 0;
    size_t capacity = 64, length = 0;
    char *text = malloc(capacity);
    if (!text) {
      tensor_free(&output);
      break;
    }
    text[0] = '\0';
    for (int step = 0; step < steps; ++step) {
      const float *row = output.data + (size_t)step * classes;
      int index = 0;
      for (int c = 1; c < classes; ++c) {
        if (row[c] > row[index]) index = c;
      }
      if (index && index != previous && (size_t)index < engine->character_count) {
        const char *character = engine->characters[index];
        const size_t needed = length + strlen(character) + 1;
        if (needed > capacity) {
          capacity = needed * 2;
          char *grown = realloc(text, capacity);
          if (!grown) break;
          text = grown;
        }
        strcat(text, character);
        length = strlen(text);
        score_sum += row[index];
        ++score_count;
      }
      previous = index;
    }
    tensor_free(&output);
    results[i].text = text;
    results[i].score = score_count ? score_sum / score_count : 0;
  }
  for (size_t i = 0; i < count; ++i) {
    if (!results[i].text) results[i].text = strdup("");
  }
  *lines = results;
  return 0;
}

static char *engine_run(Engine *engine, const unsigned char *data, size_t size) {
  Image decoded;
  if (decode_image(data, size, &decoded) != 0) {
    snprintf(g_error, sizeof g_error, "unsupported image");
    return NULL;
  }
  Image bounded, image;
  if (resize_within_bounds(&decoded, 30, 2000, &bounded) != 0) {
    image_free(&decoded);
    return NULL;
  }
  image_free(&decoded);
  if (vertical_padding(&bounded, &image) != 0) {
    image_free(&bounded);
    return NULL;
  }
  image_free(&bounded);

  Box *boxes = NULL;
  size_t box_count = 0;
  if (detect(engine, &image, &boxes, &box_count) != 0) {
    image_free(&image);
    return NULL;
  }
  Image *crops = calloc(box_count ? box_count : 1, sizeof(Image));
  size_t crop_count = 0;
  if (!crops) {
    free(boxes);
    image_free(&image);
    snprintf(g_error, sizeof g_error, "out of memory");
    return NULL;
  }
  for (size_t i = 0; i < box_count; ++i) {
    Image cropped;
    const int status = crop_box(&image, &boxes[i], &cropped);
    if (status < 0) {
      free(boxes);
      image_free(&image);
      free(crops);
      return NULL;
    }
    if (status > 0) continue;
    crops[crop_count++] = cropped;
  }
  free(boxes);
  image_free(&image);

  Line *lines = NULL;
  if (recognize(engine, crops, crop_count, &lines) != 0) {
    for (size_t i = 0; i < crop_count; ++i) image_free(&crops[i]);
    free(crops);
    return NULL;
  }
  for (size_t i = 0; i < crop_count; ++i) image_free(&crops[i]);
  free(crops);

  size_t capacity = 256, length = 0;
  char *text = malloc(capacity);
  if (!text) {
    for (size_t i = 0; i < crop_count; ++i) free(lines[i].text);
    free(lines);
    snprintf(g_error, sizeof g_error, "out of memory");
    return NULL;
  }
  text[0] = '\0';
  int emitted = 0;
  for (size_t i = 0; i < crop_count; ++i) {
    const char *line = lines[i].text;
    if (strspn(line, " \t\r\n") == strlen(line) || lines[i].score < .5f) continue;
    const size_t needed = length + strlen(line) + (emitted ? 2 : 1);
    if (needed > capacity) {
      capacity = needed * 2;
      char *grown = realloc(text, capacity);
      if (!grown) break;
      text = grown;
    }
    if (emitted) text[length++] = '\n';
    memcpy(text + length, line, strlen(line));
    length += strlen(line);
    text[length] = '\0';
    emitted = 1;
  }
  for (size_t i = 0; i < crop_count; ++i) free(lines[i].text);
  free(lines);
  return text;
}

// ---------------------------------------------------------------- exported ABI

void *ocr_create(const char *model_dir, char **error) {
  if (error) *error = NULL;
  if (!g_ort) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
      if (error) *error = strdup("ONNX Runtime C API unavailable");
      return NULL;
    }
    if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ocr-daemon-c", &g_env) != NULL) {
      if (error) *error = strdup("cannot create ONNX Runtime environment");
      return NULL;
    }
  }
  Engine *engine = calloc(1, sizeof(Engine));
  if (!engine) {
    if (error) *error = strdup("out of memory");
    return NULL;
  }
  char path[4096];
  snprintf(path, sizeof path, "%s/PP-OCRv6_det_small.onnx", model_dir);
  if (session_create(&engine->det, path) != 0) goto failed;
  snprintf(path, sizeof path, "%s/PP-OCRv6_rec_small.onnx", model_dir);
  if (session_create(&engine->rec, path) != 0) goto failed;
  if (load_characters(engine) != 0) goto failed;
  return engine;

failed:
  if (error) *error = strdup(g_error);
  engine_free(engine);
  return NULL;
}

char *ocr_run(void *handle, const unsigned char *data, size_t size, char **error) {
  Engine *engine = handle;
  if (error) *error = NULL;
  g_error[0] = '\0';
  char *text = engine_run(engine, data, size);
  if (!text && error) *error = strdup(g_error[0] ? g_error : "ocr failed");
  return text;
}

void ocr_destroy(void *handle) { engine_free(handle); }

void ocr_free(void *value) { free(value); }
