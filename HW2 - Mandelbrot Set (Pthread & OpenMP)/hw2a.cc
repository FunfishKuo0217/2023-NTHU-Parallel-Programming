#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <xmmintrin.h>

#define MAX_THREADS 666
int QUEUE, THREAD;
int *image;
int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;

int tasks = 0, done = 0;
pthread_mutex_t lock;

typedef struct threadpool_t threadpool_t;
threadpool_t *threadpool_create(int thread_count, int queue_size, int flags);
int threadpool_add(threadpool_t *pool, void (*routine)(void *), void *arg, int flags);
int threadpool_destroy(threadpool_t *pool, int flags);

typedef enum {
    threadpool_invalid        = -1,
    threadpool_lock_failure   = -2,
    threadpool_queue_full     = -3,
    threadpool_shutdown       = -4,
    threadpool_thread_failure = -5
} threadpool_error_t;

typedef enum {
    threadpool_graceful       = 1
} threadpool_destroy_flags_t;

typedef enum {
    immediate_shutdown = 1,
    graceful_shutdown  = 2
} threadpool_shutdown_t;

typedef struct {
    void (*function)(void *);
    void *argument;
} threadpool_task_t;


/* by row assign task */
void MSet(void *arg_j){
    int *j;
    j = (int *) arg_j;
    for(int i = 0; i < width; i++){
        double y0 = *j * ((upper - lower) / height) + lower;
        double x0 = i * ((right - left) / width) + left;
        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        // pthread_mutex_lock(&lock);
        image[*j * width + i] = repeats;
    }
}

void MSet_SSE(void *arg_j){
    int *j;
    j = (int *) arg_j;
    // __m128 j = _mm_set_ps1(*j);
    __m128d xmin = _mm_set_pd1(left);
    __m128d ymin = _mm_set_pd1(lower);
    __m128d xscale = _mm_set_pd1((right - left) / width);
    __m128d yscale = _mm_set_pd1((upper - lower) / height);
    __m128d threshold = _mm_set_pd1(4);
    __m128d one = _mm_set_pd1(1);
    
    for(int i = 0; i < width; i += 2){
        __m128d mx = _mm_set_pd(i + 0, i + 1);
        __m128d my = _mm_set_pd1(*j);
        // __m128d cr = _mm_add_pd(_mm_mul_pd(mx, xscale), xmin);
        // __m128d ci = _mm_add_pd(_mm_mul_pd(my, yscale), ymin);
        __m128d x0 = _mm_add_pd(_mm_mul_pd(mx, xscale), xmin);
        __m128d y0 = _mm_add_pd(_mm_mul_pd(my, yscale), ymin);
        // __m128d zr = cr;
        // __m128d zi = ci;
        __m128d x = _mm_set_pd1(0);
        __m128d y = _mm_set_pd1(0);
        int k = 0;
        __m128d mk = _mm_set_pd1(k);
        while(++k < iters){
            __m128d x2 = _mm_mul_pd(x, x);
            __m128d y2 = _mm_mul_pd(y, y);
            __m128d temp = _mm_add_pd(_mm_sub_pd(x2, y2), x0);
            y = _mm_add_pd(_mm_add_pd(_mm_mul_pd(x, y), _mm_mul_pd(x, y)), y0);
            x = temp;
            __m128d lenSqr = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
            __m128d mask = _mm_cmplt_pd(lenSqr, threshold);
            mk = _mm_add_pd(_mm_and_pd(mask, one), mk);
            /* Early bailout? */
            if (_mm_movemask_pd(mask) == 0)
                break;

        }
        // int pixels = _mm_cvtsd_f64(mk);
        double *pixels;
        pixels = (double*)malloc(sizeof(double)*2);
        _mm_store_pd(pixels, mk);
        // unsigned char *dst = image + y * s->width * 3 + x * 3;
        // double *src = (double *)&pixels;
        for(int s = 0; s < 2; s++){
            image[(*j) * width + i + s] = (int)pixels[s];
        }
    }
}

struct threadpool_t {
  pthread_mutex_t lock;
  pthread_cond_t notify;
  pthread_t *threads;
  threadpool_task_t *queue;
  int thread_count;
  int queue_size;
  int head;
  int tail;
  int count;
  int shutdown;
  int started;
};

static void *threadpool_thread(void *threadpool);
int threadpool_free(threadpool_t *pool);

threadpool_t *threadpool_create(int thread_count, int queue_size, int flags)
{
    threadpool_t *pool;
    int i;
    (void) flags;

    if(thread_count <= 0 || thread_count > MAX_THREADS || queue_size <= 0) {
        return NULL;
    }

    if((pool = (threadpool_t *)malloc(sizeof(threadpool_t))) == NULL) {
        goto err;
    }

    /* Initialize */
    pool->thread_count = 0;
    pool->queue_size = queue_size;
    pool->head = pool->tail = pool->count = 0;
    pool->shutdown = pool->started = 0;

    /* Allocate thread and task queue */
    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * thread_count);
    pool->queue = (threadpool_task_t *)malloc
        (sizeof(threadpool_task_t) * queue_size);

    /* Initialize mutex and conditional variable first */
    if((pthread_mutex_init(&(pool->lock), NULL) != 0) ||
       (pthread_cond_init(&(pool->notify), NULL) != 0) ||
       (pool->threads == NULL) ||
       (pool->queue == NULL)) {
        goto err;
    }

    /* Start worker threads */
    for(i = 0; i < thread_count; i++) {
        if(pthread_create(&(pool->threads[i]), NULL,
                          threadpool_thread, (void*)pool) != 0) {
            threadpool_destroy(pool, 0);
            return NULL;
        }
        //printf("thread id: %d\n", i);
        pool->thread_count++;
        pool->started++;
    }

    return pool;

 err:
    if(pool) {
        threadpool_free(pool);
    }
    return NULL;
}

/* add task in thread pool*/
int threadpool_add(threadpool_t *pool, void (*function)(void *), void *argument, int flags)
{
    int err = 0;
    int next;
    (void) flags;

    if(pool == NULL || function == NULL) {
        return threadpool_invalid;
    }

    if(pthread_mutex_lock(&(pool->lock)) != 0) {
        return threadpool_lock_failure;
    }

    next = (pool->tail + 1) % pool->queue_size;

    do {
        /* Are we full ? */
        if(pool->count == pool->queue_size) {
            err = threadpool_queue_full;
            break;
        }

        /* Are we shutting down ? */
        if(pool->shutdown) {
            err = threadpool_shutdown;
            break;
        }

        /* Add task to queue */
        pool->queue[pool->tail].function = function;
        pool->queue[pool->tail].argument = argument;
        pool->tail = next;
        pool->count += 1;

        /* pthread_cond_broadcast */
        if(pthread_cond_signal(&(pool->notify)) != 0) {
            err = threadpool_lock_failure;
            break;
        }
    } while(0);

    if(pthread_mutex_unlock(&pool->lock) != 0) {
        err = threadpool_lock_failure;
    }

    return err;
}

int threadpool_destroy(threadpool_t *pool, int flags)
{
    int i, err = 0;

    if(pool == NULL) {
        return threadpool_invalid;
    }

    if(pthread_mutex_lock(&(pool->lock)) != 0) {
        return threadpool_lock_failure;
    }

    do {
        /* Already shutting down */
        if(pool->shutdown) {
            err = threadpool_shutdown;
            break;
        }

        pool->shutdown = (flags & threadpool_graceful) ?
            graceful_shutdown : immediate_shutdown;

        /* Wake up all worker threads */
        if((pthread_cond_broadcast(&(pool->notify)) != 0) ||
           (pthread_mutex_unlock(&(pool->lock)) != 0)) {
            err = threadpool_lock_failure;
            break;
        }

        /* Join all worker thread */
        for(i = 0; i < pool->thread_count; i++) {
            if(pthread_join(pool->threads[i], NULL) != 0) {
                err = threadpool_thread_failure;
            }
        }
    } while(0);

    /* Only if everything went well do we deallocate the pool */
    if(!err) {
        threadpool_free(pool);
    }
    return err;
}

int threadpool_free(threadpool_t *pool)
{
    if(pool == NULL || pool->started > 0) {
        return -1;
    }

    /* Did we manage to allocate ? */
    if(pool->threads) {
        free(pool->threads);
        free(pool->queue);
 
        pthread_mutex_lock(&(pool->lock));
        pthread_mutex_destroy(&(pool->lock));
        pthread_cond_destroy(&(pool->notify));
    }
    free(pool);    
    return 0;
}


static void *threadpool_thread(void *threadpool)
{
    threadpool_t *pool = (threadpool_t *)threadpool;
    threadpool_task_t task;

    for(;;) {
        /* Lock must be taken to wait on conditional variable */
        pthread_mutex_lock(&(pool->lock));

        /* Wait on condition variable, check for spurious wakeups.
           When returning from pthread_cond_wait(), we own the lock. */
        while((pool->count == 0) && (!pool->shutdown)) {
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        if(pool->count == 0){break;}
        /* Grab our task */
        task.function = pool->queue[pool->head].function;
        task.argument = pool->queue[pool->head].argument;
        pool->head = (pool->head + 1) % pool->queue_size;
        pool->count -= 1;

        /* Unlock */
        pthread_mutex_unlock(&(pool->lock));

        /* Get to work */
        (*(task.function))(task.argument);
    }

    pool->started--;

    pthread_mutex_unlock(&(pool->lock));
    pthread_exit(NULL);
    return(NULL);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv)
{
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    THREAD = CPU_COUNT(&cpu_set);
    QUEUE = width*height;
    printf("%d %d\n", THREAD, QUEUE);
    threadpool_t *pool;
    pthread_mutex_init(&lock, NULL);
    int *arg_set;
    arg_set = (int *)malloc(sizeof(int)* QUEUE);
    assert((pool = threadpool_create(THREAD, QUEUE, 0)) != NULL);
    
    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    int taskid = 0;
    for(int j = 0; j < height; j++){
        arg_set[taskid] = j;
        pthread_mutex_lock(&lock);
        threadpool_add(pool, &MSet, (void*)&arg_set[taskid], 0);
        taskid++;
        pthread_mutex_unlock(&lock);
    }

    threadpool_destroy(pool, 0);
    // /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    return 0;
}