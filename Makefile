CC = gcc
# -O3: 开启高级优化
# -mavx2: 启用 AVX2 指令集
# -Wall: 开启警告
CFLAGS = -O3 -mavx2 -Wall

TARGET = benchmark_cycles

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) main.c -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)