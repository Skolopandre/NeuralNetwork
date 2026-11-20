#ifndef ALIGNED_ALLOC_H
#define ALIGNED_ALLOC_H

#include <stdlib.h>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)

#include <malloc.h>

static inline void *aligned_malloc(size_t size, size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

static inline void aligned_free(void *ptr)
{
    _aligned_free(ptr);
}

#else

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif
#include <stdlib.h>

static inline void *aligned_malloc(size_t size, size_t alignment)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return NULL;
    return ptr;
}

static inline void aligned_free(void *ptr)
{
    free(ptr);
}

#endif

#endif