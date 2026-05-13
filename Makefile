BINARY := dictate
WHISPER_DIR := whisper.cpp
WHISPER_BUILD := $(WHISPER_DIR)/build
GGML_VULKAN ?= OFF
CPU_COUNT := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
WHISPER_LIBS := libs/libggml-base.a libs/libggml-cpu.a libs/libggml.a libs/libwhisper.a
ifeq ($(GGML_VULKAN),ON)
WHISPER_LIBS += libs/libggml-vulkan.a
GO_TAGS := -tags vulkan
endif
ifeq ($(shell uname -s),Darwin)
WHISPER_LIBS += libs/libggml-metal.a libs/libggml-blas.a
endif

.PHONY: all clean whisper_libs

all: whisper_libs
	CGO_CFLAGS="-I$(CURDIR)/$(WHISPER_DIR)/include -I$(CURDIR)/$(WHISPER_DIR)/ggml/include -I$(CURDIR)/$(WHISPER_DIR)/ggml/src" \
		go build -o $(BINARY) $(GO_TAGS) .

whisper_libs:
	@missing=0; \
	for lib in $(WHISPER_LIBS); do \
		if [ ! -f "$$lib" ]; then missing=1; break; fi; \
	done; \
	if [ "$$missing" = "1" ]; then \
		echo "Building whisper.cpp libraries..."; \
		cmake -S $(WHISPER_DIR) -B $(WHISPER_BUILD) \
			-DCMAKE_BUILD_TYPE=Release \
			-DGGML_VULKAN=$(GGML_VULKAN) \
			-DBUILD_SHARED_LIBS=OFF \
			$(if $(filter Darwin,$(shell uname -s)),-DGGML_METAL=ON -DGGML_BLAS=ON); \
		cmake --build $(WHISPER_BUILD) --parallel $(CPU_COUNT); \
		mkdir -p libs; \
		cp $(WHISPER_BUILD)/src/libwhisper.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/src/whisper.lib libs/libwhisper.a; \
		cp $(WHISPER_BUILD)/ggml/src/libggml.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml.a libs/libggml.a 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml.lib libs/libggml.a; \
		cp $(WHISPER_BUILD)/ggml/src/libggml-base.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-base.a libs/libggml-base.a 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-base.lib libs/libggml-base.a; \
		cp $(WHISPER_BUILD)/ggml/src/libggml-cpu.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-cpu.a libs/libggml-cpu.a 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-cpu.lib libs/libggml-cpu.a; \
		if [ "$(GGML_VULKAN)" = "ON" ]; then \
			cp $(WHISPER_BUILD)/ggml/src/ggml-vulkan/libggml-vulkan.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-vulkan/ggml-vulkan.a libs/libggml-vulkan.a 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-vulkan/ggml-vulkan.lib libs/libggml-vulkan.a; \
		fi; \
		if [ "$(shell uname -s)" = "Darwin" ]; then \
			cp $(WHISPER_BUILD)/ggml/src/ggml-metal/libggml-metal.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-metal/ggml-metal.a libs/libggml-metal.a 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-metal/ggml-metal.lib libs/libggml-metal.a; \
			cp $(WHISPER_BUILD)/ggml/src/ggml-blas/libggml-blas.a libs/ 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-blas/ggml-blas.a libs/libggml-blas.a 2>/dev/null || cp $(WHISPER_BUILD)/ggml/src/ggml-blas/ggml-blas.lib libs/libggml-blas.a; \
		fi \
	fi

clean:
	rm -f $(BINARY) $(BINARY).exe libs/*.a libs/*.lib
	rm -rf $(WHISPER_BUILD)
