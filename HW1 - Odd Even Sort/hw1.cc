#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <mpi.h>
#include <fstream>
#define ODD 1
#define EVEN 2
int N; // element 個數
int parti; // 除 last process 以外，其餘 node 擁有的 element 個數
int read_index;
float *tmp;

// 取前半進行合併
void Merge_low(float *data, int data_Cnt, float *neighbor, int neighbor_Cnt, bool *All_Sorted){
    if (data[data_Cnt - 1] <= neighbor[0])
        return;
    int data_idx = 0, neighbor_idx = 0, tmp_index = 0;
    // 將比較後的 data 放入 tmp_buf
    while(tmp_index < data_Cnt){
        if(data_idx < data_Cnt && neighbor_idx < neighbor_Cnt){
            if(data[data_idx] <= neighbor[neighbor_idx]){
                tmp[tmp_index++] = data[data_idx++];
            }
            else{ tmp[tmp_index++] = neighbor[neighbor_idx++];}
        }
        else{
            if(data_idx >= data_Cnt){tmp[tmp_index++] = neighbor[neighbor_idx++];}
            else{tmp[tmp_index++] = data[data_idx++];}
        }
    }
    for(int i = 0; i < data_Cnt; i++){ data[i] = tmp[i]; } // 寫回 data
    return;
}
// 取後半進行合併
void Merge_high(float *data, int data_Cnt, float *neighbor, int neighbor_Cnt, bool *All_Sorted){
    // 有可能要處理 last node 問題
    if (data[0] >= neighbor[neighbor_Cnt - 1])
        return;
    int data_idx = data_Cnt-1, neighbor_idx = neighbor_Cnt-1, tmp_index = data_Cnt-1;
    while(tmp_index >= 0){
        if(data_idx >= 0 && neighbor_idx >= 0){
            if(data[data_idx] > neighbor[neighbor_idx]){
                tmp[tmp_index--] = data[data_idx--];
            }
            else{ tmp[tmp_index--] = neighbor[neighbor_idx--];}
        }
        else{
            if(data_idx < 0){
                tmp[tmp_index--] = neighbor[neighbor_idx--];
            }
            else{tmp[tmp_index--] = data[data_idx--];}
        }
    }
    for(int i = 0; i < data_Cnt; i++){ data[i] = tmp[i]; }
    return;
}

int Element_Count(int rank){
    if (parti*rank >= N){ return 0; }                      // wasted node
    if (parti*(rank+1) >= N){ return N - rank*parti; }     // true last node
    else{ return parti; }                                  // normal node
}

int main(int argc, char **argv){
    // Declaration
    double start, finish;
    MPI_Init(&argc, &argv);
    // double io_elapsed = 0, comm_elapsed = 0, computation_elapsed = 0;
    // double elapsed; // total time
    // double io_start, io_finish;
    // double comm_start, comm_finish;
    // double computation_start, computation_finish; 
    
    start = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    N = atoi(argv[1]);                     // total elementsd in the array: N
    // parti = std::ceil(N/(double)size);
    parti = N/size + 1;
    read_index = parti * rank;             // 要讀寫哪一個位置
    // double *io_buf = (double*) malloc(size*sizeof(double));
    // double *com_buf = (double*) malloc(size*sizeof(double));
    // double *comput_buf = (double*) malloc(size*sizeof(double));
    
    MPI_File input_file, output_file;
    char *input_filename = argv[2];        // input file
    char *output_filename = argv[3];       // output file

    // Create local data structure & buffer for process array by given rank
    int Cnt = Element_Count(rank);
    float *data = (float*) malloc(Cnt*sizeof(float));
    float *buf = (float*) malloc(parti*sizeof(float));
    bool isSorted = true;
    bool bufSorted = true;


    // 依照指定位置讀取檔案（if last node, 則僅能讀取 count 個 element）
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (Cnt != 0){
        MPI_File_read_at(input_file, sizeof(float) * read_index, data, Cnt, MPI_FLOAT, MPI_STATUS_IGNORE);}
    MPI_File_close(&input_file);

    

    if (Cnt != 0){ 
        // first local sort（使用 spreadsort 加速：參考 github）
        boost::sort::spreadsort::spreadsort(data, data + Cnt); 
        tmp = (float*) malloc(Cnt*sizeof(float));
    }
    // computation_finish = MPI_Wtime();
    // computation_elapsed += computation_finish - computation_start;
    // get neighbor count
    int leftCnt, rightCnt;
    if (rank-1 >= 0){leftCnt = Element_Count(rank-1);}
    else{ leftCnt = 0;}
    if (rank+1 > size){rightCnt = 0;}
    else { rightCnt = Element_Count(rank+1);}

    int iter = 0; // iteration 次數，一但超過 process 數量表示 sort 完成
    // bool lastProc = read_index  N && rank*(parti+1) >= N; 
    bool lastProc = read_index <= N && read_index + parti >= N;
    while(true){
        if(Cnt != 0){
            // Even Phase - main node (even: 0, 2, 4, 6, 8)
            if(rank%2 == 0){
                if(!lastProc){
                    // 確認是否兩邊需要交換（把 main node 最大的交換過去）                    
                    MPI_Sendrecv(&data[Cnt-1], 1, MPI_FLOAT, rank+1, EVEN, buf, 1, MPI_FLOAT, rank+1, EVEN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[Cnt-1] > buf[0]){
                        MPI_Sendrecv(data, Cnt, MPI_FLOAT, rank+1, EVEN, buf, rightCnt, MPI_FLOAT, rank+1, EVEN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_low(data, Cnt, buf, rightCnt, &isSorted);    
                    }
                }
            }
            // Even Phase - supplement node (odd: 1, 3, 5, 7, 9)
            else{
                MPI_Sendrecv(data, 1, MPI_FLOAT, rank-1, EVEN, buf, 1, MPI_FLOAT, rank-1, EVEN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[0] < buf[0]){
                    MPI_Sendrecv(data, Cnt, MPI_FLOAT, rank-1, EVEN, buf, leftCnt, MPI_FLOAT, rank-1, EVEN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    Merge_high(data, Cnt, buf, leftCnt, &isSorted);
                }
            }

            // Odd Phase - main node (odd: 1, 3, 5, 7, 9) - 和右邊的比較
            if(rank%2 == 1){
                if(!lastProc){
                    MPI_Sendrecv(&data[Cnt-1], 1, MPI_FLOAT, rank+1, ODD, buf, 1, MPI_FLOAT, rank+1, ODD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[Cnt-1] > buf[0]){
                        MPI_Sendrecv(data, Cnt, MPI_FLOAT, rank+1, ODD, buf, rightCnt, MPI_FLOAT, rank+1, ODD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_low(data, Cnt, buf, rightCnt, &isSorted);
                    }
                }
            }
            // Odd Phase - supplement node (odd: 0, 2, 4, 6, 8)
            else{
                if(rank != 0){ // rank 0 沒有左邊 process 可以進行交換
                    MPI_Sendrecv(data, 1, MPI_FLOAT, rank-1, ODD, buf, 1, MPI_FLOAT, rank-1, ODD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[0] < buf[0]){
                        MPI_Sendrecv(data, Cnt, MPI_FLOAT, rank-1, ODD, buf, leftCnt, MPI_FLOAT, rank-1, ODD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_high(data, Cnt, buf, leftCnt, &isSorted);
                    }
                }
            }   
        }
    
        iter += 2;
        if(iter > size){break;}
    }
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * read_index, data, Cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    free(data);
    free(buf);
    if (Cnt != 0)(free(tmp));
    return 0;
}

