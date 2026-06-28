UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else ifneq (,$(findstring MINGW,$(UNAME_S)))
    PLATFORM := windows
else ifneq (,$(findstring MSYS,$(UNAME_S)))
    PLATFORM := windows
else
    $(error Unsupported platform: $(UNAME_S))
endif

MAKEFILE := Makefile.$(PLATFORM)

.PHONY: all clean whisper_libs install uninstall help

all clean whisper_libs install uninstall:
	@echo "==> Building on $(PLATFORM) using $(MAKEFILE)"
	$(MAKE) -f $(MAKEFILE) $@

help:
	@echo "Detected platform: $(PLATFORM) ($(UNAME_S))"
	@echo "Targets: all | clean | whisper_libs | help"
	@echo "Overrides (forwarded): GGML_VULKAN=ON|OFF"
	@echo "Direct: make -f $(MAKEFILE)"
