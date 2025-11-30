#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h> // AVX2
#include <x86intrin.h> // rdtsc

#define DATA_SIZE (1 << 20)
#define WARMUP_ROUNDS 1000
#define TEST_ROUNDS 10000

// ---------------------------------------------------------
// 计时工具: RDTSC
// ---------------------------------------------------------
// 使用 lfence 防止指令重排，确保测量准确
static inline uint64_t read_tsc_start()
{
    _mm_lfence();
    return __rdtsc();
}

static inline uint64_t read_tsc_end()
{
    _mm_lfence(); // 序列化指令流
    unsigned int aux;
    return __rdtscp(&aux); // 读取并获取处理器ID（这里主要用它的序列化特性）
}

// ---------------------------------------------------------
// 方法 1: AVX2 SIMD (每次处理 4 个 u64)
// ---------------------------------------------------------
void add_avx2(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t n)
{
    for (size_t i = 0; i < n; i += 4)
    {
        __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
        __m256i vb = _mm256_load_si256((const __m256i *)&b[i]);
        __m256i vr = _mm256_add_epi64(va, vb);
        _mm256_store_si256((__m256i *)&result[i], vr);
    }
}

// ---------------------------------------------------------
// 2. AVX2 - Version 2: 循环 8 次展开
// ---------------------------------------------------------
// 这里的“8次展开”指展开 8 条 AVX 指令。
// 1 条 AVX 指令处理 4 个 u64，所以一次循环处理 8 * 4 = 32 个元素。
void add_avx2_v2(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t n)
{
    for (size_t i = 0; i < n; i += 32)
    {
        // Load
        __m256i va0 = _mm256_load_si256((const __m256i *)&a[i + 0]);
        __m256i va1 = _mm256_load_si256((const __m256i *)&a[i + 4]);
        __m256i va2 = _mm256_load_si256((const __m256i *)&a[i + 8]);
        __m256i va3 = _mm256_load_si256((const __m256i *)&a[i + 12]);
        __m256i va4 = _mm256_load_si256((const __m256i *)&a[i + 16]);
        __m256i va5 = _mm256_load_si256((const __m256i *)&a[i + 20]);
        __m256i va6 = _mm256_load_si256((const __m256i *)&a[i + 24]);
        __m256i va7 = _mm256_load_si256((const __m256i *)&a[i + 28]);

        __m256i vb0 = _mm256_load_si256((const __m256i *)&b[i + 0]);
        __m256i vb1 = _mm256_load_si256((const __m256i *)&b[i + 4]);
        __m256i vb2 = _mm256_load_si256((const __m256i *)&b[i + 8]);
        __m256i vb3 = _mm256_load_si256((const __m256i *)&b[i + 12]);
        __m256i vb4 = _mm256_load_si256((const __m256i *)&b[i + 16]);
        __m256i vb5 = _mm256_load_si256((const __m256i *)&b[i + 20]);
        __m256i vb6 = _mm256_load_si256((const __m256i *)&b[i + 24]);
        __m256i vb7 = _mm256_load_si256((const __m256i *)&b[i + 28]);

        // Add
        __m256i vr0 = _mm256_add_epi64(va0, vb0);
        __m256i vr1 = _mm256_add_epi64(va1, vb1);
        __m256i vr2 = _mm256_add_epi64(va2, vb2);
        __m256i vr3 = _mm256_add_epi64(va3, vb3);
        __m256i vr4 = _mm256_add_epi64(va4, vb4);
        __m256i vr5 = _mm256_add_epi64(va5, vb5);
        __m256i vr6 = _mm256_add_epi64(va6, vb6);
        __m256i vr7 = _mm256_add_epi64(va7, vb7);

        // Store
        _mm256_store_si256((__m256i *)&result[i + 0], vr0);
        _mm256_store_si256((__m256i *)&result[i + 4], vr1);
        _mm256_store_si256((__m256i *)&result[i + 8], vr2);
        _mm256_store_si256((__m256i *)&result[i + 12], vr3);
        _mm256_store_si256((__m256i *)&result[i + 16], vr4);
        _mm256_store_si256((__m256i *)&result[i + 20], vr5);
        _mm256_store_si256((__m256i *)&result[i + 24], vr6);
        _mm256_store_si256((__m256i *)&result[i + 28], vr7);
    }
}

// ---------------------------------------------------------
// 3. AVX2 - Version 3: 循环 4 次展开
// ---------------------------------------------------------
// 这里的“4次展开”指展开 4 条 AVX 指令。
// 1 条 AVX 指令处理 4 个 u64，所以一次循环处理 4 * 4 = 16 个元素。
void add_avx2_v3(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t n)
{
    for (size_t i = 0; i < n; i += 16)
    {
        // Load
        __m256i va0 = _mm256_load_si256((const __m256i *)&a[i + 0]);
        __m256i va1 = _mm256_load_si256((const __m256i *)&a[i + 4]);
        __m256i va2 = _mm256_load_si256((const __m256i *)&a[i + 8]);
        __m256i va3 = _mm256_load_si256((const __m256i *)&a[i + 12]);

        __m256i vb0 = _mm256_load_si256((const __m256i *)&b[i + 0]);
        __m256i vb1 = _mm256_load_si256((const __m256i *)&b[i + 4]);
        __m256i vb2 = _mm256_load_si256((const __m256i *)&b[i + 8]);
        __m256i vb3 = _mm256_load_si256((const __m256i *)&b[i + 12]);

        // Add
        __m256i vr0 = _mm256_add_epi64(va0, vb0);
        __m256i vr1 = _mm256_add_epi64(va1, vb1);
        __m256i vr2 = _mm256_add_epi64(va2, vb2);
        __m256i vr3 = _mm256_add_epi64(va3, vb3);

        // Store
        _mm256_store_si256((__m256i *)&result[i + 0], vr0);
        _mm256_store_si256((__m256i *)&result[i + 4], vr1);
        _mm256_store_si256((__m256i *)&result[i + 8], vr2);
        _mm256_store_si256((__m256i *)&result[i + 12], vr3);
    }
}

// ---------------------------------------------------------
// 方法 2: Scalar Version 1 (循环不展开)
// ---------------------------------------------------------
// 强制禁止自动向量化(no-tree-vectorize)和自动循环展开(no-unroll-loops)
// 确保测试的是最原始的循环性能
__attribute__((optimize("no-tree-vectorize", "no-unroll-loops"))) void add_scalar_v1(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        result[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------
// 方法 3: Scalar Version 2 (循环8次展开)
// ---------------------------------------------------------
// 强制禁止自动向量化，但允许我们手动展开
__attribute__((optimize("no-tree-vectorize"))) void add_scalar_v2_unroll8(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t n)
{
    // 假设 n 是 8 的倍数 (2^15 肯定是)
    for (size_t i = 0; i < n; i += 8)
    {
        result[i + 0] = a[i + 0] + b[i + 0];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
        result[i + 4] = a[i + 4] + b[i + 4];
        result[i + 5] = a[i + 5] + b[i + 5];
        result[i + 6] = a[i + 6] + b[i + 6];
        result[i + 7] = a[i + 7] + b[i + 7];
    }
}

// ---------------------------------------------------------
// 测试与统计辅助函数
// ---------------------------------------------------------
typedef void (*test_func)(const uint64_t *, const uint64_t *, uint64_t *, size_t);

void benchmark(const char *name, test_func func,
               const uint64_t *a, const uint64_t *b, uint64_t *res)
{

    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;

    // 预热 Cache
    for (int i = 0; i < WARMUP_ROUNDS; i++)
    {
        func(a, b, res, DATA_SIZE);
    }

    // 正式测试
    for (int i = 0; i < TEST_ROUNDS; i++)
    {
        uint64_t start = read_tsc_start();
        func(a, b, res, DATA_SIZE);
        uint64_t end = read_tsc_end();

        uint64_t cycles = end - start;
        total_cycles += cycles;
        if (cycles < min_cycles)
            min_cycles = cycles;
    }

    double avg_cycles = (double)total_cycles / TEST_ROUNDS;
    double cpe = avg_cycles / DATA_SIZE; // Cycles Per Element

    printf("%-25s | Avg Cycles: %10.0f | Min Cycles: %10lu | CPE: %.4f\n",
           name, avg_cycles, min_cycles, cpe);
}

int main()
{
    printf("Benchmark: Array Size = %d (u64)\n", DATA_SIZE);
    printf("Metrics: CPU Cycles (RDTSC)\n");
    printf("-------------------------------------------------------------------------\n");

    // 内存对齐分配 (AVX2 需要 32 字节对齐)
    size_t size_bytes = DATA_SIZE * sizeof(uint64_t);
    uint64_t *a = (uint64_t *)_mm_malloc(size_bytes, 32);
    uint64_t *b = (uint64_t *)_mm_malloc(size_bytes, 32);
    uint64_t *res = (uint64_t *)_mm_malloc(size_bytes, 32);

    // 初始化数据
    for (size_t i = 0; i < DATA_SIZE; i++)
    {
        a[i] = i * 13;
        b[i] = i * 7;
    }

    benchmark("Scalar (No Unroll)", add_scalar_v1, a, b, res);
    benchmark("Scalar (Unroll 8)", add_scalar_v2_unroll8, a, b, res);
    benchmark("AVX2 SIMD", add_avx2, a, b, res);
    benchmark("AVX2 SIMD (Unroll 8)", add_avx2_v2, a, b, res);
    benchmark("AVX2 SIMD (Unroll 4)", add_avx2_v3, a, b, res);

    _mm_free(a);
    _mm_free(b);
    _mm_free(res);

    return 0;
}