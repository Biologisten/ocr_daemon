// Shared crop geometry. The implementations here are the ones the C core was
// built and measured with (see README.md): reproducing OpenCV's fixed-point
// linear resize, its min-area rectangle and its bicubic warp is what keeps the
// recognised text close to the reference pipeline, and both cores now use these
// exact functions so their crops cannot drift apart.
#define _POSIX_C_SOURCE 200809L

#include "ocr_geometry.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ERROR_SIZE 256
#define COEF_BITS 11
#define COEF_SCALE (1 << COEF_BITS)

static char g_error[ERROR_SIZE];

static int fail(const char *message) {
  snprintf(g_error, sizeof g_error, "%s", message);
  return -1;
}

const char *ocr_geometry_error(void) { return g_error; }

int ocr_py_round(double value) {
  double lower = floor(value), fraction = value - lower;
  if (fraction < .5)
    return (int)lower;
  if (fraction > .5)
    return (int)lower + 1;
  return (int)lower % 2 == 0 ? (int)lower : (int)lower + 1;
}

int ocr_resize_side(double value) { return ocr_py_round(value / 32.0) * 32; }

// RapidOCR's limit_side_len, and the budget that keeps a strip screenshot from
// inflating it. The cap only binds on strip shapes: 1100x159 would otherwise
// take a 5088x736 canvas (3.7 Mpx). 3.0 Mpx was chosen by sweeping the budget
// against the corpus' text: 3.6 Mpx leaves every image's text byte-identical
// but gives back most of the saving, and 2.5 Mpx starts splitting lines. At 3.0
// the detector canvas of 19.png is 4544x640 instead of 5088x736.
#define DETECTOR_SHORT_SIDE 736
#define DETECTOR_MAX_PIXELS 3000000.0

// Number of CPUs in a sibling list like "0,1" or "0-3,8-11".
static int sibling_count(const char *list) {
  int count = 0;
  const char *at = list;
  while (*at >= '0' && *at <= '9') {
    char *end = NULL;
    const long first = strtol(at, &end, 10);
    if (end == at)
      break;
    long last = first;
    at = end;
    if (*at == '-') {
      last = strtol(at + 1, &end, 10);
      if (end == at + 1)
        break;
      at = end;
    }
    if (last >= first)
      count += (int)(last - first + 1);
    if (*at != ',')
      break;
    ++at;
  }
  return count;
}

int ocr_intra_op_threads(void) {
  const long logical = sysconf(_SC_NPROCESSORS_ONLN);
  if (logical <= 1)
    return 0;
  FILE *file =
      fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
  if (!file)
    return 0;
  char text[256] = {0};
  const int siblings = fgets(text, sizeof text, file) ? sibling_count(text) : 0;
  fclose(file);
  if (siblings <= 1)
    return 0; // no SMT: the runtime's own default is the right one
  const long physical = logical / siblings;
  return physical >= 1 && physical < logical ? (int)physical : 0;
}

void ocr_detection_canvas(int height, int width, int *rows, int *cols) {
  const int smallest = height < width ? height : width;
  const double ratio = smallest < DETECTOR_SHORT_SIDE
                           ? (double)DETECTOR_SHORT_SIDE / smallest
                           : 1.0;
  int r = ocr_resize_side(height * ratio), c = ocr_resize_side(width * ratio);
  if (r < 32)
    r = 32;
  if (c < 32)
    c = 32;
  const double area = (double)r * (double)c;
  if (area > DETECTOR_MAX_PIXELS) {
    // Floor to the 32-px grid so the budget stays a hard upper bound.
    const double shrink = sqrt(DETECTOR_MAX_PIXELS / area);
    const int shrunk_rows = (int)(r * shrink) / 32 * 32;
    const int shrunk_cols = (int)(c * shrink) / 32 * 32;
    *rows = shrunk_rows < 32 ? 32 : shrunk_rows;
    *cols = shrunk_cols < 32 ? 32 : shrunk_cols;
    return;
  }
  *rows = r;
  *cols = c;
}

double ocr_distance(OcrPoint a, OcrPoint b) {
  return hypot(a.x - b.x, a.y - b.y);
}

// Horizontal half of the resize for one source row: the exact fixed-point sums
// the fused per-pixel loop would compute for column x, channel c.
static void resize_row(const unsigned char *row, const int *firsts,
                       const int *seconds, const short *xcoef, int dst_cols,
                       int *out) {
  for (int x = 0; x < dst_cols; ++x) {
    const int c0 = xcoef[2 * x], c1 = xcoef[2 * x + 1];
    const unsigned char *p0 = row + firsts[x], *p1 = row + seconds[x];
    int *target = out + x * 3;
    for (int c = 0; c < 3; ++c)
      target[c] = p0[c] * c0 + p1[c] * c1;
  }
}

// cv::resize(INTER_LINEAR) on 8-bit BGR: 11-bit fixed-point coefficients, a
// horizontal pass into int, then a vertical pass with rounding.
void ocr_resize_bilinear(const unsigned char *src, int src_rows, int src_cols,
                         unsigned char *dst, int dst_rows, int dst_cols) {
  if (src_rows == dst_rows && src_cols == dst_cols) {
    memcpy(dst, src, (size_t)src_rows * src_cols * 3);
    return;
  }
  const double scale_x = (double)src_cols / dst_cols;
  const double scale_y = (double)src_rows / dst_rows;
  int *firsts = malloc(sizeof(int) * (size_t)dst_cols);
  int *seconds = malloc(sizeof(int) * (size_t)dst_cols);
  short *xcoef = malloc(sizeof(short) * 2 * (size_t)dst_cols);
  if (!firsts || !seconds || !xcoef) {
    free(firsts);
    free(seconds);
    free(xcoef);
    memset(dst, 0, (size_t)dst_rows * dst_cols * 3);
    return;
  }
  for (int x = 0; x < dst_cols; ++x) {
    double dx = (x + 0.5) * scale_x - 0.5;
    int sx = (int)floor(dx);
    double fx = dx - sx;
    if (sx < 0) {
      fx = 0;
      sx = 0;
    } else if (sx >= src_cols - 1) {
      fx = 0;
      sx = src_cols - 1;
    }
    const int sx2 = sx + 1 < src_cols ? sx + 1 : src_cols - 1;
    firsts[x] = sx * 3;
    seconds[x] = sx2 * 3;
    xcoef[2 * x] = (short)ocr_py_round((1.0 - fx) * COEF_SCALE);
    xcoef[2 * x + 1] = (short)ocr_py_round(fx * COEF_SCALE);
  }

  // The horizontal pass depends only on the source row, and neighbouring output
  // rows usually share both of theirs, so it is computed once per source row
  // instead of once per output row. The arithmetic is unchanged, so the result
  // is bit-identical to the fused form.
  const size_t row_values = (size_t)dst_cols * 3;
  int *sums[2] = {malloc(row_values * sizeof(int)),
                  malloc(row_values * sizeof(int))};
  if (!sums[0] || !sums[1]) {
    free(sums[0]);
    free(sums[1]);
    free(firsts);
    free(seconds);
    free(xcoef);
    memset(dst, 0, (size_t)dst_rows * dst_cols * 3);
    return;
  }
  int cached[2] = {-1, -1};

  for (int y = 0; y < dst_rows; ++y) {
    double dy = (y + 0.5) * scale_y - 0.5;
    int sy = (int)floor(dy);
    double fy = dy - sy;
    if (sy < 0) {
      fy = 0;
      sy = 0;
    } else if (sy >= src_rows - 1) {
      fy = 0;
      sy = src_rows - 1;
    }
    const int sy2 = sy + 1 < src_rows ? sy + 1 : src_rows - 1;
    const int ycoef0 = (short)ocr_py_round((1.0 - fy) * COEF_SCALE);
    const int ycoef1 = (short)ocr_py_round(fy * COEF_SCALE);
    for (int row = 0; row < 2; ++row) {
      const int source = row ? sy2 : sy;
      if (cached[row] == source)
        continue;
      resize_row(src + (size_t)source * src_cols * 3, firsts, seconds, xcoef,
                 dst_cols, sums[row]);
      cached[row] = source;
    }
    const int *top = sums[0], *bottom = sums[1];
    unsigned char *out = dst + (size_t)y * dst_cols * 3;
    for (size_t i = 0; i < row_values; ++i) {
      const int value =
          (top[i] * ycoef0 + bottom[i] * ycoef1 + (1 << (2 * COEF_BITS - 1))) >>
          (2 * COEF_BITS);
      out[i] = (unsigned char)((unsigned)value <= 255u ? value
                               : value < 0             ? 0
                                                       : 255);
    }
  }
  free(sums[0]);
  free(sums[1]);
  free(firsts);
  free(seconds);
  free(xcoef);
}

static double cubic_weight(double x) {
  const double a = -0.75; // OpenCV's INTER_CUBIC coefficient
  x = fabs(x);
  if (x <= 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0)
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

static int round_half_even(double value) {
  const double lower = floor(value), fraction = value - lower;
  if (fraction < .5)
    return (int)lower;
  if (fraction > .5)
    return (int)lower + 1;
  return (int)lower % 2 == 0 ? (int)lower : (int)lower + 1;
}

void ocr_warp_cubic(const unsigned char *src, int src_rows, int src_cols,
                    unsigned char *dst, int rows, int cols,
                    const double matrix[9]) {
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      unsigned char *out = dst + ((size_t)y * cols + x) * 3;
      const double w = matrix[6] * x + matrix[7] * y + matrix[8];
      if (fabs(w) < 1e-12) {
        out[0] = out[1] = out[2] = 0;
        continue;
      }
      const double u = (matrix[0] * x + matrix[1] * y + matrix[2]) / w;
      const double v = (matrix[3] * x + matrix[4] * y + matrix[5]) / w;
      const int iu = (int)floor(u) - 1, iv = (int)floor(v) - 1;
      double wx[4], wy[4];
      int xs[4], ys[4];
      for (int i = 0; i < 4; ++i) {
        wx[i] = cubic_weight(u - (iu + i));
        wy[i] = cubic_weight(v - (iv + i));
        const int sx = iu + i, sy = iv + i;
        xs[i] = sx < 0 ? 0 : sx >= src_cols ? src_cols - 1 : sx;
        ys[i] = sy < 0 ? 0 : sy >= src_rows ? src_rows - 1 : sy;
      }
      double sum[3] = {0, 0, 0};
      for (int dy = 0; dy < 4; ++dy) {
        if (wy[dy] == 0)
          continue;
        const unsigned char *row = src + (size_t)ys[dy] * src_cols * 3;
        for (int dx = 0; dx < 4; ++dx) {
          const double weight = wy[dy] * wx[dx];
          if (weight == 0)
            continue;
          const unsigned char *pixel = row + xs[dx] * 3;
          sum[0] += weight * pixel[0];
          sum[1] += weight * pixel[1];
          sum[2] += weight * pixel[2];
        }
      }
      for (int c = 0; c < 3; ++c) {
        const int value = round_half_even(sum[c]);
        out[c] = (unsigned char)(value < 0 ? 0 : value > 255 ? 255 : value);
      }
    }
  }
}

void ocr_nchw_normalize(float *out, const unsigned char *bgr, int cols,
                        int target_h, int target_w) {
  const int plane = target_h * target_w;
  for (int y = 0; y < target_h; ++y) {
    for (int x = 0; x < cols; ++x) {
      const unsigned char *pixel = bgr + ((size_t)y * cols + x) * 3;
      for (int c = 0; c < 3; ++c)
        out[c * plane + y * target_w + x] =
            ((float)pixel[c] / 255.0f - .5f) / .5f;
    }
  }
}

unsigned char *ocr_detection_mask(const float *probability, int rows,
                                  int cols) {
  const size_t size = (size_t)rows * (size_t)cols;
  unsigned char *mask = malloc(size);
  unsigned char *dilated = calloc(size, 1);
  if (!mask || !dilated) {
    free(mask);
    free(dilated);
    return NULL;
  }
  for (size_t i = 0; i < size; ++i)
    mask[i] = probability[i] > 0.3f ? 255 : 0;
  // cv::dilate with a 2x2 kernel uses anchor (0,0): out(x,y)=max over (x..x+1,
  // y..y+1).
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      unsigned char value = mask[(size_t)y * cols + x];
      if (x + 1 < cols && mask[(size_t)y * cols + x + 1] > value)
        value = mask[(size_t)y * cols + x + 1];
      if (y + 1 < rows && mask[(size_t)(y + 1) * cols + x] > value)
        value = mask[(size_t)(y + 1) * cols + x];
      if (x + 1 < cols && y + 1 < rows &&
          mask[(size_t)(y + 1) * cols + x + 1] > value) {
        value = mask[(size_t)(y + 1) * cols + x + 1];
      }
      dilated[(size_t)y * cols + x] = value;
    }
  }
  free(mask);
  return dilated;
}

void ocr_groups_free(OcrPixelGroups *groups) {
  for (size_t i = 0; i < groups->count; ++i) {
    free(groups->items[i].x);
    free(groups->items[i].y);
  }
  free(groups->items);
  groups->items = NULL;
  groups->count = 0;
}

int ocr_group_pixels(const unsigned char *mask, int rows, int cols,
                     unsigned char value, int eight_connected, int skip_border,
                     OcrPixelGroups *groups) {
  unsigned char *visited = calloc((size_t)rows * (size_t)cols, 1);
  int *stack = malloc(sizeof(int) * (size_t)rows * (size_t)cols);
  groups->items = NULL;
  groups->count = 0;
  if (!visited || !stack) {
    free(visited);
    free(stack);
    return fail("out of memory");
  }
  const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1},
            dy8[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  const int neighbours = eight_connected ? 8 : 4;
  const int matches = (value != 0);
  size_t capacity = 0;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const size_t start = (size_t)y * cols + x;
      if (visited[start] || (mask[start] != 0) != matches)
        continue;
      size_t top = 0;
      stack[top++] = (int)start;
      visited[start] = 1;
      OcrPixels pixels = {NULL, NULL, 0};
      const size_t first = groups->count;
      if (first + 1 > capacity) {
        capacity = capacity ? capacity * 2 : 64;
        OcrPixels *grown = realloc(groups->items, capacity * sizeof(OcrPixels));
        if (!grown) {
          free(visited);
          free(stack);
          groups->items = NULL;
          groups->count = 0;
          return fail("out of memory");
        }
        groups->items = grown;
      }
      int on_border = 0;
      size_t collected = 0, pixel_capacity = 0;
      while (top > 0) {
        const int current = stack[--top];
        const int cy = current / cols, cx = current % cols;
        if (cy == 0 || cx == 0 || cy == rows - 1 || cx == cols - 1)
          on_border = 1;
        // Only boundary pixels matter for the convex hull, and skipping the
        // interior keeps minAreaRect's sort small on large text blobs.
        const int interior = cx > 0 && cx < cols - 1 && cy > 0 &&
                             cy < rows - 1 &&
                             ((mask[current - 1] != 0) == matches) &&
                             ((mask[current + 1] != 0) == matches) &&
                             ((mask[current - cols] != 0) == matches) &&
                             ((mask[current + cols] != 0) == matches);
        if (!interior) {
          if (collected == pixel_capacity) {
            const size_t next = pixel_capacity ? pixel_capacity * 2 : 64;
            int *grown_x = realloc(pixels.x, next * sizeof(int));
            int *grown_y =
                grown_x ? realloc(pixels.y, next * sizeof(int)) : NULL;
            if (!grown_x || !grown_y) {
              free(grown_x ? grown_x : pixels.x);
              free(grown_y ? grown_y : pixels.y);
              free(visited);
              free(stack);
              groups->items = NULL;
              groups->count = 0;
              return fail("out of memory");
            }
            pixels.x = grown_x;
            pixels.y = grown_y;
            pixel_capacity = next;
          }
          pixels.x[collected] = cx;
          pixels.y[collected] = cy;
          ++collected;
        }
        for (int i = 0; i < neighbours; ++i) {
          const int ny = cy + dy8[i], nx = cx + dx8[i];
          if (ny < 0 || nx < 0 || ny >= rows || nx >= cols)
            continue;
          const size_t next = (size_t)ny * cols + nx;
          if (visited[next])
            continue;
          if ((mask[next] != 0) != matches)
            continue;
          visited[next] = 1;
          stack[top++] = (int)next;
        }
      }
      pixels.count = collected;
      if ((skip_border && on_border) || collected == 0) {
        free(pixels.x);
        free(pixels.y);
        continue;
      }
      groups->items[first] = pixels;
      ++groups->count;
    }
  }
  free(visited);
  free(stack);
  return 0;
}

static int compare_by_x(const void *a, const void *b) {
  double left = ((const OcrPoint *)a)->x, right = ((const OcrPoint *)b)->x;
  return left < right ? -1 : left > right ? 1 : 0;
}

static int compare_by_y(const void *a, const void *b) {
  double left = ((const OcrPoint *)a)->y, right = ((const OcrPoint *)b)->y;
  return left < right ? -1 : left > right ? 1 : 0;
}

void ocr_order_box(const OcrPoint raw[4], OcrBox *box) {
  OcrPoint sorted[4];
  memcpy(sorted, raw, sizeof sorted);
  qsort(sorted, 4, sizeof(OcrPoint), compare_by_x);
  OcrPoint left[2] = {sorted[0], sorted[1]}, right[2] = {sorted[2], sorted[3]};
  qsort(left, 2, sizeof(OcrPoint), compare_by_y);
  qsort(right, 2, sizeof(OcrPoint), compare_by_y);
  box->c[0] = left[0];
  box->c[1] = right[0];
  box->c[2] = right[1];
  box->c[3] = left[1];
}

static double cross(OcrPoint origin, OcrPoint a, OcrPoint b) {
  return (a.x - origin.x) * (b.y - origin.y) -
         (a.y - origin.y) * (b.x - origin.x);
}

// Andrew's monotone chain; returns the hull size (capacity must be 2 * count).
static size_t convex_hull(OcrPoint *points, size_t count, OcrPoint *hull) {
  if (count < 3)
    return 0;
  qsort(points, count, sizeof(OcrPoint), compare_by_x);
  size_t k = 0;
  for (size_t i = 0; i < count; ++i) {
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], points[i]) <= 0)
      --k;
    hull[k++] = points[i];
  }
  for (size_t i = count - 1, lower = k + 1; i-- > 0;) {
    while (k >= lower && cross(hull[k - 2], hull[k - 1], points[i]) <= 0)
      --k;
    hull[k++] = points[i];
  }
  return k - 1; // the closing point repeats the first
}

int ocr_min_area_rect(OcrPoint *points, size_t count, OcrRect *rect) {
  if (count == 0)
    return fail("empty contour");
  OcrPoint *hull = malloc(sizeof(OcrPoint) * 2 * count);
  if (!hull)
    return fail("out of memory");
  const size_t size = convex_hull(points, count, hull);
  if (size < 3) {
    // Fewer than 3 hull points: cv::minAreaRect yields a zero-width rect, which
    // the caller's "min side < 3" filter discards. Report it the same way.
    memset(rect, 0, sizeof *rect);
    free(hull);
    return 0;
  }
  double best_area = -1;
  for (size_t edge = 0; edge < size; ++edge) {
    const OcrPoint a = hull[edge], b = hull[(edge + 1) % size];
    double dx = b.x - a.x, dy = b.y - a.y;
    const double length = hypot(dx, dy);
    if (length < 1e-9)
      continue;
    dx /= length;
    dy /= length;
    double min_u = 1e30, max_u = -1e30, min_v = 1e30, max_v = -1e30;
    for (size_t i = 0; i < size; ++i) {
      const double u = hull[i].x * dx + hull[i].y * dy;
      const double v = -hull[i].x * dy + hull[i].y * dx;
      if (u < min_u)
        min_u = u;
      if (u > max_u)
        max_u = u;
      if (v < min_v)
        min_v = v;
      if (v > max_v)
        max_v = v;
    }
    const double area = (max_u - min_u) * (max_v - min_v);
    if (best_area < 0 || area < best_area) {
      best_area = area;
      const double center_u = (min_u + max_u) / 2,
                   center_v = (min_v + max_v) / 2;
      rect->ux = dx;
      rect->uy = dy;
      rect->vx = -dy;
      rect->vy = dx;
      rect->width = max_u - min_u;
      rect->height = max_v - min_v;
      rect->cx = center_u * dx - center_v * dy;
      rect->cy = center_u * dy + center_v * dx;
    }
  }
  free(hull);
  return best_area < 0 ? fail("degenerate contour") : 0;
}

void ocr_rect_points(const OcrRect *rect, OcrPoint out[4]) {
  const double hw = rect->width / 2, hh = rect->height / 2;
  const double dxu[4] = {-hw, hw, hw, -hw}, dyu[4] = {-hh, -hh, hh, hh};
  for (int i = 0; i < 4; ++i) {
    out[i].x = rect->cx + dxu[i] * rect->ux + dyu[i] * rect->vx;
    out[i].y = rect->cy + dxu[i] * rect->uy + dyu[i] * rect->vy;
  }
}

float ocr_box_score(const float *probability, int width, int height,
                    const OcrPoint box[4]) {
  double xmin = box[0].x, xmax = box[0].x, ymin = box[0].y, ymax = box[0].y;
  for (int i = 1; i < 4; ++i) {
    xmin = box[i].x < xmin ? box[i].x : xmin;
    xmax = box[i].x > xmax ? box[i].x : xmax;
    ymin = box[i].y < ymin ? box[i].y : ymin;
    ymax = box[i].y > ymax ? box[i].y : ymax;
  }
  int y0 = (int)floor(ymin), y1 = (int)ceil(ymax), x0 = (int)floor(xmin),
      x1 = (int)ceil(xmax);
  y0 = y0 < 0 ? 0 : y0 > height - 1 ? height - 1 : y0;
  y1 = y1 < 0 ? 0 : y1 > height - 1 ? height - 1 : y1;
  x0 = x0 < 0 ? 0 : x0 > width - 1 ? width - 1 : x0;
  x1 = x1 < 0 ? 0 : x1 > width - 1 ? width - 1 : x1;
  double total = 0;
  long long count = 0;
  for (int y = y0; y <= y1; ++y) {
    const double center = y + 0.5;
    double left = 1e30, right = -1e30;
    for (int i = 0; i < 4; ++i) {
      const OcrPoint a = box[i], b = box[(i + 1) % 4];
      if ((a.y <= center && b.y > center) || (b.y <= center && a.y > center)) {
        const double t = (center - a.y) / (b.y - a.y);
        const double x = a.x + t * (b.x - a.x);
        left = x < left ? x : left;
        right = x > right ? x : right;
      }
    }
    if (left > right)
      continue;
    int xa = (int)ceil(left - 0.5), xb = (int)floor(right - 0.5);
    xa = xa < x0 ? x0 : xa > x1 ? x1 : xa;
    xb = xb < x0 ? x0 : xb > x1 ? x1 : xb;
    for (int x = xa; x <= xb; ++x) {
      total += probability[(size_t)y * width + x];
      ++count;
    }
  }
  return count ? (float)(total / (double)count) : 0.0f;
}

int ocr_perspective_transform(const OcrPoint src[4], const OcrPoint dst[4],
                              double matrix[9]) {
  double augmented[8][9];
  for (int i = 0; i < 4; ++i) {
    const double x = src[i].x, y = src[i].y, u = dst[i].x, v = dst[i].y;
    double *row0 = augmented[2 * i], *row1 = augmented[2 * i + 1];
    const double first[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
    const double second[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
    memcpy(row0, first, sizeof first);
    memcpy(row1, second, sizeof second);
  }
  for (int column = 0; column < 8; ++column) {
    int pivot = column;
    for (int row = column + 1; row < 8; ++row) {
      if (fabs(augmented[row][column]) > fabs(augmented[pivot][column]))
        pivot = row;
    }
    if (fabs(augmented[pivot][column]) < 1e-12)
      return fail("degenerate perspective transform");
    if (pivot != column) {
      for (int i = 0; i < 9; ++i) {
        const double swap = augmented[column][i];
        augmented[column][i] = augmented[pivot][i];
        augmented[pivot][i] = swap;
      }
    }
    for (int row = 0; row < 8; ++row) {
      if (row == column)
        continue;
      const double factor = augmented[row][column] / augmented[column][column];
      if (factor == 0)
        continue;
      for (int i = column; i < 9; ++i)
        augmented[row][i] -= factor * augmented[column][i];
    }
  }
  for (int i = 0; i < 8; ++i)
    matrix[i] = augmented[i][8] / augmented[i][i];
  matrix[8] = 1.0;
  return 0;
}

void ocr_sort_boxes(OcrBox *boxes, size_t count) {
  int *lines = malloc(sizeof(int) * count);
  if (!lines)
    return;
  for (size_t i = 1; i < count; ++i) { // stable insertion sort by top edge
    const OcrBox current = boxes[i];
    size_t j = i;
    while (j > 0 && boxes[j - 1].c[0].y > current.c[0].y) {
      boxes[j] = boxes[j - 1];
      --j;
    }
    boxes[j] = current;
  }
  for (size_t i = 0; i < count; ++i)
    lines[i] =
        i ? lines[i - 1] + (boxes[i].c[0].y - boxes[i - 1].c[0].y >= 10 ? 1 : 0)
          : 0;
  for (size_t i = 1; i < count; ++i) {
    const OcrBox current = boxes[i];
    const int line = lines[i];
    size_t j = i;
    while (j > 0 &&
           (lines[j - 1] > line ||
            (lines[j - 1] == line && boxes[j - 1].c[0].x > current.c[0].x))) {
      boxes[j] = boxes[j - 1];
      lines[j] = lines[j - 1];
      --j;
    }
    boxes[j] = current;
    lines[j] = line;
  }
  free(lines);
}
