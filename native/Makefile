CC ?= cc
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra -I/usr/include/opencv5 -I/usr/include/onnxruntime
CFLAGS := -O3 -std=c11 -Wall -Wextra
OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_geometry -lopencv_core

# Use the vendored ONNX Runtime (see lib/NOTICE.md) when it is present; fall back
# to the system library otherwise so a fresh checkout still builds.
ORT_VERSION := 1.31.1
ORT_DIR := lib
ORT_LIB := $(ORT_DIR)/libonnxruntime.so.$(ORT_VERSION)

ifeq ($(wildcard $(ORT_LIB)),)
  ORT_CXX := -lonnxruntime
  ORT_NAME := system ONNX Runtime
else
  ORT_CXX := -L$(ORT_DIR) -l:libonnxruntime.so.$(ORT_VERSION) -Wl,-rpath,'$$ORIGIN/$(ORT_DIR)'
  ORT_NAME := vendored ONNX Runtime $(ORT_VERSION)
endif

LDLIBS := $(OPENCV_LIBS) -lpthread

all: c-ocr-daemon cpp-ocr-daemon
	@echo "linked against $(ORT_NAME)"

# Crop geometry shared by both cores (see ocr_geometry.h), compiled once here
# and linked into each archive.
ocr_geometry.o: ocr_geometry.c ocr_geometry.h
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

ocr_core.o: ocr_core.cpp ocr_core.h ocr_geometry.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

libocr_core.a: ocr_core.o ocr_geometry.o
	ar rcs $@ $^

# The C candidate is C11 end to end: its frontend and its OCR core, which uses
# the ONNX Runtime C API, libpng and libjpeg instead of OpenCV.
c_daemon.o: c_daemon.c ocr_core.h
	$(CC) $(CFLAGS) -c $< -o $@

ocr_core_c.o: ocr_core_c.c ocr_core.h ocr_geometry.h
	$(CC) $(CFLAGS) -c $< -o $@

libocr_core_c.a: ocr_core_c.o ocr_geometry.o
	ar rcs $@ $^

c-ocr-daemon: c_daemon.o libocr_core_c.a
	$(CXX) $(CXXFLAGS) $< -L. -locr_core_c $(ORT_CXX) -lpng -ljpeg -lm -lpthread -o $@

cpp-ocr-daemon: cpp_daemon.cpp libocr_core.a
	$(CXX) $(CXXFLAGS) $< -L. -locr_core $(ORT_CXX) $(LDLIBS) -o $@

smoke: smoke.cpp libocr_core.a
	$(CXX) $(CXXFLAGS) $< -L. -locr_core $(ORT_CXX) $(LDLIBS) -o $@

# Self-contained tree: the binary resolves $ORIGIN/lib, so the vendored runtime
# sits beside it and the system 1.29 is never consulted.
PREFIX ?= $(HOME)/.local/lib/ocr-daemon

install: cpp-ocr-daemon
	mkdir -p $(PREFIX)/lib $(PREFIX)/models
	install -m755 cpp-ocr-daemon $(PREFIX)/ocr-daemon
	install -m755 $(ORT_LIB) $(PREFIX)/lib/
	ln -sf libonnxruntime.so.$(ORT_VERSION) $(PREFIX)/lib/libonnxruntime.so.1
	install -m644 ../models/*.onnx $(PREFIX)/models/

clean:
	rm -f *.o *.a c-ocr-daemon cpp-ocr-daemon smoke
