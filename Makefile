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

.PHONY: all clean whisper_libs

all: whisper_libs
	CGO_CFLAGS="-I$(CURDIR)/$(WHISPER_DIR)/include -I$(CURDIR)/$(WHISPER_DIR)/ggml/include" \
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
			-DBUILD_SHARED_LIBS=OFF; \
		cmake --build $(WHISPER_BUILD) --parallel $(CPU_COUNT); \
		mkdir -p libs; \
		cp $(WHISPER_BUILD)/src/libwhisper.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/libggml.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/libggml-base.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/libggml-cpu.a libs/; \
		if [ "$(GGML_VULKAN)" = "ON" ]; then \
			cp $(WHISPER_BUILD)/ggml/src/ggml-vulkan/libggml-vulkan.a libs/; \
		fi \
	fi

clean:
	rm -f $(BINARY) libs/*.a
	rm -rf $(WHISPER_BUILD)
