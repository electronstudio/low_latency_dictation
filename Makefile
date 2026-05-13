BINARY := dictate
WHISPER_DIR := whisper.cpp
WHISPER_BUILD := $(WHISPER_DIR)/build
WHISPER_LIBS := libs/libggml-base.a libs/libggml-cpu.a libs/libggml-vulkan.a libs/libggml.a libs/libwhisper.a

.PHONY: all clean whisper_libs

all: whisper_libs
	CGO_CFLAGS="-I$(CURDIR)/$(WHISPER_DIR)/include -I$(CURDIR)/$(WHISPER_DIR)/ggml/include" \
		go build -o $(BINARY) .

whisper_libs:
	@missing=0; \
	for lib in $(WHISPER_LIBS); do \
		if [ ! -f "$$lib" ]; then missing=1; break; fi; \
	done; \
	if [ "$$missing" = "1" ]; then \
		echo "Building whisper.cpp libraries..."; \
		cmake -S $(WHISPER_DIR) -B $(WHISPER_BUILD) \
			-DCMAKE_BUILD_TYPE=Release \
			-DGGML_VULKAN=ON \
			-DBUILD_SHARED_LIBS=OFF; \
		cmake --build $(WHISPER_BUILD) --parallel $$(nproc); \
		mkdir -p libs; \
		cp $(WHISPER_BUILD)/src/libwhisper.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/libggml.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/libggml-base.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/libggml-cpu.a libs/; \
		cp $(WHISPER_BUILD)/ggml/src/ggml-vulkan/libggml-vulkan.a libs/; \
	fi

clean:
	rm -f $(BINARY) $(WHISPER_LIBS)
	rm -rf $(WHISPER_BUILD)
