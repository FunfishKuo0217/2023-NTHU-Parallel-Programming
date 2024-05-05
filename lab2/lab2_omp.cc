#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
unsigned long long r;
unsigned long long k;
unsigned long long r_sqr;
unsigned long long ans;

int main(int argc, char** argv) {
    unsigned long long pixels;
    unsigned long long omp_threads, tid;
    if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
    r = atoll(argv[1]);
	k = atoll(argv[2]);
    ans = 0;
    r_sqr = r*r;
    // unsigned long long x;
    pixels = 0;
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

#pragma omp parallel reduction(+:ans) private(tid) shared(ncpus)
    {
        pixels = 0;
        // omp_threads = omp_get_num_threads();
        tid = omp_get_thread_num();
        // printf("Hello: thread %llu/%llu\n", omp_thread, omp_threads);
        for(unsigned long long x = tid; x < r; x += ncpus){
            // printf("Calculate %llu/%llu\n", x*x, tid);
            ans += ceil(sqrtl(r_sqr - x*x));
        }
    }
    ans = ans%k;    
    printf("%llu\n", (4 * ans) % k);
    return 0;
}
