#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
unsigned long long r_sqr;
unsigned long long partition;

int main(int argc, char** argv) {
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long ans = 0;
	unsigned long long partial_ans = 0;
	r_sqr = r*r;
	MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks;
	unsigned long long omp_threads, omp_thread;
	
    char hostname[HOST_NAME_MAX];

    assert(!gethostname(hostname, HOST_NAME_MAX));
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	// unsigned long long partition = r/mpi_ranks;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	// 資源超額: n > r (不用動用其他資源)
	if (r <= mpi_ranks){
		partition = r;
		if (mpi_rank < r){
			partial_ans = 0;
			unsigned long long start = mpi_rank; // process start
			// unsigned long long end = start + partition; // process end
			for (unsigned long long x = start; x < r; x+=partition) {
				unsigned long long y = ceil(sqrtl(r_sqr - x*x));
				partial_ans += y;
				partial_ans %= k;
			}
			//printf("working process %llu %llu %llu\n", start, mpi_rank, partial_ans); 
		}
		MPI_Reduce(&partial_ans, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		if (mpi_rank == 0)
			printf("%llu\n", (4 * ans) % k);
		MPI_Finalize();
	}
	else{
		partition = r/mpi_ranks;  // 一塊 process 負責的份量
		unsigned long long start = partition*mpi_rank; // process start
		unsigned long long end = start + partition; // process end
		if(mpi_ranks == mpi_rank+1  && r > end){end = r;}
		unsigned long long partial_ans = 0;
		#pragma omp parallel reduction(+:partial_ans) private(omp_thread) shared(omp_threads, start, end)
		{	
			// unsigned long long partial_ans = 0;
			omp_thread = omp_get_thread_num();
			omp_threads = omp_get_num_threads();
			// printf("Hello: thread %llu/%llu\n", omp_thread, omp_threads);
			for(unsigned long long x = omp_thread+start; x < end; x += omp_threads){
				partial_ans += ceil(sqrtl(r_sqr - x*x));
			}
		}
		partial_ans = partial_ans%k;  
		// printf("%llu ")
		MPI_Reduce(&partial_ans, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		if (mpi_rank == 0)
			printf("%llu\n", (4 * ans) % k);
		MPI_Finalize();
	}
	return 0;
} 
