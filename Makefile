CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I/usr/include/SDL2

LDFLAGS = libs/libwhisper.a libs/libggml.a libs/libggml-base.a libs/libggml-cpu.a libs/libggml-vulkan.a
LDLIBS = -lvulkan -fopenmp -lSDL2

SRCS = main.cpp common-sdl.cpp
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

TARGET = dictation

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)
