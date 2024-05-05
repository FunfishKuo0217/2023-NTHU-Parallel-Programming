#define max_thread 7500
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
unsigned long long mid;
unsigned long long ans;
unsigned long long k;
unsigned long long r;
unsigned long long r_sqr;
int num_threads;
pthread_mutex_t ans_mutex;

void *cal_subsum(void *tid){
    int i, start, *mytid, end;
    unsigned long long pixels = 0;
    /* Initialize my part of the global array and keep local sum */
    mytid = (int *) tid;
    start = *mytid;
    unsigned long long jump = (unsigned long long) num_threads;
    // end = start + mid;
    // if(num_threads == *mytid+1 && r > end){end = r;}
    for (unsigned long long x = start; x < r; x+=jump) {
        unsigned long long y = ceil(sqrtl(r_sqr - x*x));
        pixels += y; 
    }
    /* Lock the mutex and update the global sum, then exit */
    pixels %= k;
    pthread_mutex_lock (&ans_mutex);
    ans = ans + pixels;
    pthread_mutex_unlock (&ans_mutex);
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
    r_sqr = r*r;
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	int ncpus = CPU_COUNT(&cpuset);
    num_threads = ncpus;
    // mid = r/ncpus;
    ans = 0;
    // pthread_t tids[num_threads];
    pthread_t threads[num_threads];
    int ID[num_threads];
    // pthread_attr_t attr;
    
    pthread_mutex_init(&ans_mutex, NULL);
    // pthread_attr_init(&attr);
    // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for (int i = 0; i < num_threads; i++) {
        ID[i] = i;
        pthread_create(&threads[i], NULL, cal_subsum, (void *) &ID[i]);
    }
    // // printf("%d\n", num_threads);
    /* Wait for all threads to complete then print global sum */ 
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    // printf ("\n[MAIN] Done. Sum= %e", ans);
    /* Clean up and exit */
    printf("%llu\n", (4 * ans) % k);
    // pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&ans_mutex);
    pthread_exit (NULL);
    
}