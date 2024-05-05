// blocked Floyed Warshal version
#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <xmmintrin.h>
#include <vector>
#include <omp.h>
#include <limits>
#include <cstring>
#include <math.h>
#define Block_size 64
#define offset 32
#define OUR_INF 1073741823

int n, edge, n_padding;

/* read file function */
void Correctness(char *file, int* ans);

__global__ void FWP1(int* dev_dist, int round, int n_padding){
    int i = threadIdx.y;
    int j = threadIdx.x;

    // printf("(%d %d)\n", i, j);
    // 想想看如果沒有 + 1 會怎樣（就不是 pivot）
    int up_left = round*Block_size*(n_padding+1) + i*n_padding + j;
    int up_right = round*Block_size*(n_padding+1) + i*n_padding + j + offset;
    int down_left = round*Block_size*(n_padding+1) + (i+offset)*n_padding + j;
    int down_right = round*Block_size*(n_padding+1) + (i+offset)*n_padding + j + offset;

    int offset_i = i+offset;
    int offset_j = j+offset;

    __shared__ int sdata[Block_size][Block_size];
    
    sdata[i][j] = dev_dist[up_left];
    sdata[i][offset_j] = dev_dist[up_right];
    sdata[offset_i][j] = dev_dist[down_left];
    sdata[offset_i][offset_j] = dev_dist[down_right];

    __syncthreads();
    
    #pragma unroll 64
    for(int k = 0; k < Block_size; k++){
        __syncthreads();
        sdata[i][j] = min(sdata[i][j], sdata[i][k]+sdata[k][j]);
        sdata[i][offset_j] = min(sdata[i][offset_j], sdata[i][k]+sdata[k][offset_j]);
        sdata[offset_i][j] = min(sdata[offset_i][j], sdata[offset_i][k]+sdata[k][j]);
        sdata[offset_i][offset_j] = min(sdata[offset_i][offset_j], sdata[offset_i][k]+sdata[k][offset_j]);
    }

    dev_dist[up_left] = sdata[i][j];
    dev_dist[up_right] = sdata[i][offset_j];
    dev_dist[down_left] = sdata[offset_i][j];
    dev_dist[down_right] = sdata[offset_i][offset_j];
}

__global__ void FWP2(int *dev_dist, int round, int n_padding){
    // pivot block
    // 傳入的 round 指的是 第幾個 round，用來看 pivot block 是誰 
    int i = threadIdx.y;
    int j = threadIdx.x;

    int pivot_base = round*Block_size*(n_padding+1);
    int up_left = pivot_base + i*n_padding + j;
    int up_right = pivot_base + i*n_padding + j + offset;
    int down_left = pivot_base + (i+offset)*n_padding + j;
    int down_right = pivot_base + (i+offset)*n_padding + j + offset;
    
    int offset_i = i+offset;
    int offset_j = j+offset;

    __shared__ int pivot[Block_size][Block_size];
    pivot[i][j] = dev_dist[up_left];
    pivot[i][offset_j] = dev_dist[up_right];
    pivot[offset_i][j] = dev_dist[down_left];
    pivot[offset_i][offset_j] = dev_dist[down_right];

    // pivot row or column
    int block_j = blockIdx.x;
    if (block_j == round){return;}

    int h_base = round*Block_size*n_padding + block_j*Block_size;
    int v_base = block_j*Block_size*n_padding + round*Block_size;

    int h1 = h_base + i*n_padding + j;
    int h2 = h_base + i*n_padding + j + offset;
    int h3 = h_base + (i+offset)*n_padding + j;
    int h4 = h_base + (i+offset)*n_padding + j + offset;

    int v1 = v_base + i*n_padding + j;
    int v2 = v_base + i*n_padding + j + offset;
    int v3 = v_base + (i+offset)*n_padding + j;
    int v4 = v_base + (i+offset)*n_padding + j + offset;

    __shared__ int pivot_row[Block_size][Block_size];
    __shared__ int pivot_col[Block_size][Block_size];

    pivot_row[i][j] = dev_dist[h1];
    pivot_row[i][offset_j] = dev_dist[h2];
    pivot_row[offset_i][j] = dev_dist[h3];
    pivot_row[offset_i][offset_j] = dev_dist[h4];
    pivot_col[i][j] = dev_dist[v1];
    pivot_col[i][offset_j] = dev_dist[v2];
    pivot_col[offset_i][j] = dev_dist[v3];
    pivot_col[offset_i][offset_j] = dev_dist[v4];

    __syncthreads();
    #pragma unroll 64    
    for(int k = 0; k < Block_size; k++){
        __syncthreads();
        pivot_row[i][j] = min(pivot_row[i][j], pivot[i][k] + pivot_row[k][j]);
        pivot_row[i][offset_j] = min(pivot_row[i][offset_j], pivot[i][k] + pivot_row[k][offset_j]);
        pivot_row[offset_i][j] = min(pivot_row[offset_i][j], pivot[offset_i][k] + pivot_row[k][j]);
        pivot_row[offset_i][offset_j] = min(pivot_row[offset_i][offset_j], pivot[offset_i][k] + pivot_row[k][offset_j]);
        
        pivot_col[i][j] = min(pivot_col[i][j], pivot_col[i][k] + pivot[k][j]);
        pivot_col[i][offset_j] = min(pivot_col[i][offset_j], pivot_col[i][k] + pivot[k][offset_j]);
        pivot_col[offset_i][j] = min(pivot_col[offset_i][j], pivot_col[offset_i][k] + pivot[k][j]);
        pivot_col[offset_i][offset_j] = min(pivot_col[offset_i][offset_j], pivot_col[offset_i][k] + pivot[k][offset_j]);
        
    }
    
    dev_dist[h1] = pivot_row[i][j];
    dev_dist[h2] = pivot_row[i][offset_j];
    dev_dist[h3] = pivot_row[offset_i][j];
    dev_dist[h4] = pivot_row[offset_i][offset_j];
    
    dev_dist[v1] = pivot_col[i][j];
    dev_dist[v2] = pivot_col[i][offset_j];
    dev_dist[v3] = pivot_col[offset_i][j];
    dev_dist[v4] = pivot_col[offset_i][offset_j];

}


__global__ void FWP3(int *dev_dist, int round, int n_padding){
    /* 定位計算格 */
    int i = threadIdx.y;
    int j = threadIdx.x;
    int offset_i = i+offset;
    int offset_j = j+offset;

    int b_i = blockIdx.y; // block row_id
    int b_j = blockIdx.x; // blck col_id
    // 把 pivot 系列全部 returns
    if(b_i == round || b_j == round){return;}
    
    // row 和 col 是要從 pivot 的角度出發
    int h_base = (round*Block_size*n_padding) + b_j*Block_size;
    int v_base = (b_i*Block_size*n_padding) + round*Block_size;

    int h1 = h_base + i*n_padding + j;
    int h2 = h_base + i*n_padding + j + offset;
    int h3 = h_base + (i+offset)*n_padding + j;
    int h4 = h_base + (i+offset)*n_padding + j + offset;

    int v1 = v_base + i*n_padding + j;
    int v2 = v_base + i*n_padding + j + offset;
    int v3 = v_base + (i+offset)*n_padding + j;
    int v4 = v_base + (i+offset)*n_padding + j + offset;

    __shared__ int pivot_row[Block_size][Block_size];
    __shared__ int pivot_col[Block_size][Block_size];
    __shared__ int outcome[Block_size][Block_size];

    //目標的 block base
    int target_base = b_i*n_padding*Block_size + b_j*Block_size; 
    // printf("pivot = (%d %d), (%d %d)\n", round, round, b_i, b_j);
    int t1 = target_base + i*n_padding + j;
    int t2 = target_base + i*n_padding + j + offset;
    int t3 = target_base + (i+offset)*n_padding + j;
    int t4 = target_base + (i+offset)*n_padding + j + offset;

    pivot_row[i][j] = dev_dist[h1];
    pivot_row[i][offset_j] = dev_dist[h2];
    pivot_row[offset_i][j] = dev_dist[h3];
    pivot_row[offset_i][offset_j] = dev_dist[h4];

    pivot_col[i][j] = dev_dist[v1];
    pivot_col[i][offset_j] = dev_dist[v2];
    pivot_col[offset_i][j] = dev_dist[v3];
    pivot_col[offset_i][offset_j] = dev_dist[v4];

    outcome[i][j] = dev_dist[t1];
    outcome[i][offset_j] = dev_dist[t2];
    outcome[offset_i][j] = dev_dist[t3];
    outcome[offset_i][offset_j] = dev_dist[t4];
    
    __syncthreads();
    #pragma unroll 64
    for(int k = 0; k < Block_size; k++){
        outcome[i][j] = min(outcome[i][j], pivot_col[i][k] + pivot_row[k][j]);
        outcome[i][offset_j] = min(outcome[i][offset_j], pivot_col[i][k] + pivot_row[k][offset_j]);
        outcome[offset_i][j] = min(outcome[offset_i][j],  pivot_col[offset_i][k] + pivot_row[k][j]);
        outcome[offset_i][offset_j] = min(outcome[offset_i][offset_j],  pivot_col[offset_i][k] + pivot_row[k][offset_j]);
    }

    dev_dist[t1] = outcome[i][j];
    dev_dist[t2] = outcome[i][offset_j];
    dev_dist[t3] = outcome[offset_i][j];
    dev_dist[t4] = outcome[offset_i][offset_j];
}


/* 1D version */
int main(int argc, char **argv)
{
    FILE* inFile = fopen(argv[1], "rb");
    fread(&n, sizeof(int), 1, inFile);
    fread(&edge, sizeof(int), 1, inFile);
    int pair[3];
    
    /* 計算 padding 以決定方便 laod memory，如果不足 64 要補到 64 的倍數 */
    n_padding = 0;
    if(n%Block_size == 0){n_padding = n;}
    else{n_padding = n + Block_size-(n%Block_size);}
    printf("n_padding %d\n", n_padding);
    
    int *ans, *GMatrix, *out;

    // /* host memory*/
    // /* init on CPU */
    GMatrix = (int*) malloc(sizeof(int)*n_padding*n_padding);
    ans = (int*) malloc(sizeof(int)*n_padding*n_padding);
    out = (int*) malloc(sizeof(int)*n*n);
    
    for(int i = 0; i < n_padding; i++){
        for(int j = 0; j < n_padding; j++){
            if(i == j && i < n){GMatrix[i * n_padding + j] = 0;}
            else{
                GMatrix[i * n_padding + j] = OUR_INF;
            }
        }
    }
    for(int i = 0; i < edge; i++){
        fread(pair, sizeof(int), 3, inFile);
        GMatrix[pair[0]*n_padding+pair[1]] = pair[2];
    }
    // for(int i = 0; i < G.E; i++){GMatrix[G.edges[i].source*n_padding + G.edges[i].dest] = G.edges[i].weight;}
    
    /* devcie memory */
    int *dev_dist = NULL;
    size_t size_G = sizeof(int)*n_padding*n_padding;
    cudaMalloc((void **)&dev_dist, size_G);
    cudaMemcpy(dev_dist, GMatrix, size_G, cudaMemcpyHostToDevice);

    // block size 
    int blocks = (n_padding + Block_size - 1)/Block_size;
    dim3 nB1(1);
    dim3 nB2(blocks);
    dim3 nB3(blocks, blocks);
    dim3 nT(offset, offset);
    int round = n_padding/Block_size;
    
    // /* Funciton */
    for(int r = 0; r < round; r++){
        /* Phase 1 */
        FWP1<<<nB1, nT>>>(dev_dist, r, n_padding);
        /* Phase 2 */
        FWP2<<<nB2, nT>>>(dev_dist, r, n_padding);
        /* Phase 3 */
        FWP3<<<nB3, nT>>>(dev_dist, r, n_padding);
    }

    // // cudaMemcpy(...) copy result image to host
    cudaMemcpy(ans, dev_dist, size_G, cudaMemcpyDeviceToHost);

    FILE *outFile = fopen(argv[2], "wb");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            out[i*n+j] = ans[i*n_padding+j];
        }
    }
    fwrite(out, sizeof(int), n*n, outFile);
    fclose(inFile);
    fclose(outFile);
    // printf("total time: %f", end-start);
    delete[] GMatrix;
    return 0;

}


void Correctness(char *file, int* ans){
    int element;
    int wrong = 0;
    FILE* ansfile = fopen(file, "rb");
    for(int i = 0; i < n; i++){
        fread(&element, sizeof(int), 1, ansfile);
        if(element != ans[i]){
            wrong += 1;
            printf("%d %d\n", ans[i], element);
        }
    }
    printf("wrond element count = %d\n", wrong);
    return;
}