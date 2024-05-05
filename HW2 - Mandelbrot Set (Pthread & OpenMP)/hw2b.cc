#define PNG_NO_SETJMP
#define TAG 0
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <png.h>

/* Global Variables */
int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
int thread;


/*functions*/
void write_png(const char* filename, int iters, int width, int height, const int* buffer);


int main(int argc, char** argv){

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    thread = CPU_COUNT(&cpu_set);

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

    /* initialization */
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks;
    unsigned long long omp_threads, omp_thread;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
    omp_lock_t lock;
    omp_init_lock(&lock);

    /* allocate memory for image */
    int *image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    int row_proc[height] = {0};
    for(int j = 0; j < height; j++){
        row_proc[j] = j%mpi_ranks;
    }

    #pragma omp parallel num_threads(thread) shared(image)
    {   
        // start_time = omp_get_wtime();
        #pragma omp for schedule(dynamic, 1) nowait
        for(int j = 0; j < height; j++){
            /* 先處理 j 和 mpi_rank 的對應關係 */
            int responsor = row_proc[j];
            if (mpi_rank == responsor){
                // printf("process %d working\n", responsor);
                // 每個 thread 開始 dynamic 地搶工作
                for(int i = 0; i < width; i++){
                    double y0 = j * ((upper - lower) / height) + lower;
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
                    // omp_set_lock(&lock);
                    // printf("process %d working on %d\n", mpi_rank, j);
                    image[j * width + i] = repeats;
                    // omp_unset_lock(&lock);
                }
                // printf("TID %d = %f\n", tid, threadtime_sec[tid]);
            }
        }
        // int tid = omp_get_thread_num();
        // end_time = omp_get_wtime();
        // threadtime_sec[tid] += (end_time-start_time);
    }     

    // /* allocate row to process */
    // for(int j = 0; j < height; j++){
    //     /* 先處理 j 和 mpi_rank 的對應關係 */
    //     int responsor = row_proc[j];
    //     if (mpi_rank == responsor){
    //         // printf("process %d working\n", responsor);
    //         #pragma omp parallel num_threads(thread) shared(image, j)
    //         {
    //             #pragma omp for schedule(dynamic, 1) nowait
    //             for(int i = 0; i < width; i++){
    //                 double y0 = j * ((upper - lower) / height) + lower;
    //                 double x0 = i * ((right - left) / width) + left;
    //                 int repeats = 0;
    //                 double x = 0;
    //                 double y = 0;
    //                 double length_squared = 0;
    //                 while (repeats < iters && length_squared < 4) {
    //                     double temp = x * x - y * y + x0;
    //                     y = 2 * x * y + y0;
    //                     x = temp;
    //                     length_squared = x * x + y * y;
    //                     ++repeats;
    //                 }
                    
    //                 // pthread_mutex_lock(&lock);
    //                 omp_set_lock(&lock);
    //                 // printf("process %d working on %d\n", mpi_rank, j);
    //                 image[j * width + i] = repeats;
    //                 omp_unset_lock(&lock);
    //             }
    //         }
    //     } 
    // }
    omp_destroy_lock(&lock);
    
    for(int j = 0; j < height; j++){
        if(mpi_rank == row_proc[j] && mpi_rank != 0){
            MPI_Send(&image[j * width], width, MPI_INT, 0, TAG, MPI_COMM_WORLD);
        }
        if(mpi_rank == 0 && row_proc[j] != 0){
            MPI_Recv(&image[j * width], width, MPI_INT, row_proc[j], TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if(mpi_rank == 0){
        write_png(filename, iters, width, height, image);
    }
    free(image);
    /* Finish and Return */
    MPI_Finalize();
    return 0;
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

