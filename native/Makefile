CC ?= cc
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra -I/usr/include/opencv5 -I/usr/include/onnxruntime
CFLAGS := -O3 -std=c11 -Wall -Wextra
OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_geometry -lopencv_core

# Link the system ONNX Runtime when it is 1.30 or newer; otherwise fall back to
# the vendored 1.31.1 (see lib/NOTICE.md), or to the system library when there
# is no vendored copy so a fresh checkout still builds.
ORT_VERSION := 1.31.1
ORT_MIN_VERSION := 1.30
ORT_DIR := lib
ORT_LIB := $(ORT_DIR)/libonnxruntime.so.$(ORT_VERSION)

SYS_ORT_PATH := $(shell ldconfig -p 2>/dev/null | awk '/libonnxruntime\.so\.1 / {print $$NF; exit}')
SYS_ORT_VERSION := $(shell [ -n "$(SYS_ORT_PATH)" ] && basename "$$(readlink -f "$(SYS_ORT_PATH)")" | sed -E 's/^libonnxruntime\.so\.//')
SYS_ORT_OK := $(shell [ -n "$(SYS_ORT_VERSION)" ] && \
  [ "$$(printf '%s\n%s\n' $(ORT_MIN_VERSION) $(SYS_ORT_VERSION) | sort -V | head -n1)" = "$(ORT_MIN_VERSION)" ] && echo yes)

ifeq ($(SYS_ORT_OK),yes)
  ORT_CXX := -lonnxruntime
  ORT_NAME := system ONNX Runtime $(SYS_ORT_VERSION)
  ORT_VENDORED := no
else ifneq ($(wildcard $(ORT_LIB)),)
  ORT_CXX := -L$(ORT_DIR) -l:libonnxruntime.so.$(ORT_VERSION) -Wl,-rpath,'$$ORIGIN/$(ORT_DIR)'
  ORT_NAME := vendored ONNX Runtime $(ORT_VERSION)
  ORT_VENDORED := yes
else
  ORT_CXX := -lonnxruntime
  ORT_NAME := system ONNX Runtime
  ORT_VENDORED := no
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
ifeq ($(ORT_VENDORED),yes)
	install -m755 $(ORT_LIB) $(PREFIX)/lib/
	ln -sf libonnxruntime.so.$(ORT_VERSION) $(PREFIX)/lib/libonnxruntime.so.1
endif
	install -m644 ../models/*.onnx $(PREFIX)/models/

# clangd's database; regenerate after changing a flag or adding a source file.
# It holds absolute paths, so it is generated here rather than committed.
compile_commands.json: Makefile
	@printf '[\n' > $@
	@sep=0; for spec in \
	  '$(CC)|$(CFLAGS)|ocr_geometry.c' \
	  '$(CXX)|$(CXXFLAGS)|ocr_core.cpp' \
	  '$(CC)|$(CFLAGS)|c_daemon.c' \
	  '$(CC)|$(CFLAGS)|ocr_core_c.c' \
	  '$(CXX)|$(CXXFLAGS)|cpp_daemon.cpp' \
	  '$(CXX)|$(CXXFLAGS)|smoke.cpp'; do \
	  c=$${spec%%|*}; rest=$${spec#*|}; f=$${rest%%|*}; s=$${rest#*|}; \
	  [ $$sep -eq 1 ] && printf ',\n' >> $@; \
	  printf '  {"directory": "%s", "file": "%s", "command": "%s %s -c %s"}' \
	    "$(CURDIR)" "$$s" "$$c" "$$f" "$$s" >> $@; \
	  sep=1; \
	done; printf '\n]\n' >> $@
	@echo "wrote $@"

clean:
	rm -f *.o *.a c-ocr-daemon cpp-ocr-daemon smoke compile_commands.json
