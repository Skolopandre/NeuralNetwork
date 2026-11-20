#ifndef VECTOR_H
#define VECTOR_H

#include <immintrin.h>

#ifdef __AVX2__

typedef struct {
    __m256 reg;
} vecf;

#define VEC_WIDTH 8

static inline vecf vec_load(const float *ptr)
{
    return (vecf){ _mm256_load_ps(ptr) };
}

static inline void vec_store(float *ptr, vecf v)
{
    _mm256_store_ps(ptr, v.reg);
}

static inline vecf vec_set1(float x)
{
    return (vecf){ _mm256_set1_ps(x) };
}

static inline vecf vec_add(vecf a, vecf b)
{
    return (vecf){ _mm256_add_ps(a.reg, b.reg) };
}

static inline vecf vec_sub(vecf a, vecf b)
{
    return (vecf){ _mm256_sub_ps(a.reg, b.reg) };
}

static inline vecf vec_mul(vecf a, vecf b)
{
    return (vecf){ _mm256_mul_ps(a.reg, b.reg) };
}

static inline vecf vec_div(vecf a, vecf b)
{
    return (vecf){ _mm256_div_ps(a.reg, b.reg) };
}

static inline vecf vec_sqrt(vecf a)
{
    return (vecf){ _mm256_sqrt_ps(a.reg) };
}

static inline vecf vec_fma(vecf a, vecf b, vecf c)
{
    return (vecf){ _mm256_fmadd_ps(a.reg, b.reg, c.reg) };
}

static inline float dot_product(
    const float *restrict a,
    const float *restrict b,
    size_t n)
{
    size_t i = 0;
    float sum = 0.0f;

#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(va, vb)
        );
    }

    __m128 low = _mm256_castps256_ps128(acc);
    __m128 high = _mm256_extractf128_ps(acc, 1);

    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    sum = _mm_cvtss_f32(sum128);
#endif

    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

static float dot_avx2(const float *restrict w,const float *restrict x,int n){
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();

    int i = 0;
    int limit = n & ~7;

    for (; i < limit; i += 8) {
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vw, vx, acc);
    }

    __m128 low  = _mm256_castps256_ps128(acc);
    __m128 high = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    for (; i < n; i++) {
        sum += w[i] * x[i];
    }

    return sum;
#else               // Fallback scalaire si AVX2 n’est pas dispo
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += w[i] * x[i];
    }
    return sum;
#endif
}

#else

typedef struct {
    float data[1];
} vecf;

#define VEC_WIDTH 1

/* implémentation scalaire */

#endif

static inline vecf vec_mul_scalar(vecf a, float x)
{
    return vec_mul(a, vec_set1(x));
}

static inline vecf vec_add_scalar(vecf a, float x)
{
    return vec_add(a, vec_set1(x));
}

static inline vecf vec_sub_scalar(vecf a, float x)
{
    return vec_sub(a, vec_set1(x));
}

static inline vecf vec_div_scalar(vecf a, float x)
{
    return vec_div(a, vec_set1(x));
}

static inline vecf vec_fmadd_scalar(float a, vecf b, vecf c)
{
    return vec_fma(vec_set1(a), b, c);
}

#endif
