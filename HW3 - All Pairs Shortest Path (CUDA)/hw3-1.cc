// blocked Floyed Warshal version
#include <sched.h>
#include <assert.h>
#include <png.h>
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
int ncpus;

struct Edge{
    int source;
    int dest;
    int weight;
};

struct Graph{
    int V; // node count
    int E;
    std::vector<Edge> edges;
};

/* read file function */
Graph ReadG(const char *filename);


/* Floyed Warshakll */
void FW_seq(Graph *G){
    int i, j, k;
    std::vector<std::vector<int>> GMatrix(G->V, std::vector<int>(G->V, std::numeric_limits<int>::infinity()));
    /* init */
    for(i = 0; i < G->E; i++){
        GMatrix[G->edges[i].source][G->edges[i].dest] = G->edges[i].weight;
    } 
    for(i = 0; i < G->V; i++){GMatrix[i][i] = 0;}

    int V = G->V;
    printf("%d\n", V);
    for(int k = 0; k < V; k++){
        for(int i = 0; i < V; i++){
            for(int j = 0; j < V; j++){
                if (GMatrix[i][j] > (GMatrix[i][k] + GMatrix[k][j]) && GMatrix[k][j] != std::numeric_limits<int>::infinity() && GMatrix[i][k] != std::numeric_limits<int>::infinity()){
                    GMatrix[i][j] = GMatrix[i][k] + GMatrix[k][j];
                }
                    
            }
        }
    }
}


int main(int argc, char **argv)
{
    /* resource */
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
    int32_t OUR_INF = 1073741823;
    Graph G;
    const char* filename = argv[1];
    const char* output_filename = argv[2];
    G = ReadG(filename);
    
    std::vector<std::vector<int>> GMatrix(G.V, std::vector<int>(G.V, OUR_INF));
    /* init */
    #pragma omp parallel for
    for(int i = 0; i < G.E; i++){
        GMatrix[G.edges[i].source][G.edges[i].dest] = G.edges[i].weight;
    } 
    /* omp version */
    #pragma omp parallel for
    for(int i = 0; i < G.V; i++){GMatrix[i][i] = 0;}
    int i, j = 0;
    #pragma omp parallel num_threads(ncpus) shared(GMatrix) private(i, j)
    {
        for(int k = 0; k < G.V; k++){   
            #pragma omp for schedule(dynamic)
            for(i = 0; i < G.V; i++){
                if(GMatrix[i][k] != OUR_INF){
                    for(j = 0; j < G.V; j++){
                        if (GMatrix[i][j] > (GMatrix[i][k] + GMatrix[k][j])){
                            // printf("%d %d %d\n", i, j, k);
                            GMatrix[i][j] = GMatrix[i][k] + GMatrix[k][j];
                        }
                    }
                }
            }
        }
    }


    /* Write File */
    std::ofstream output;
    output.open(output_filename, std::ofstream::binary);
    for(int i = 0; i < G.V; i++){
        for(int j = 0; j < G.V; j++){
            output.write((char*)&GMatrix[i][j], sizeof(int32_t));
        }
    }
    return 0;

}


Graph ReadG(const char *filename){
    Graph G;
    std::fstream file;
    file.open(filename, std::ios_base::binary | std::ios_base::in);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
    int32_t V, E;
    file.read ((char*)&V, sizeof(int32_t));
    file.read ((char*)&E, sizeof(int32_t));
    G.V = V;
    G.E = E;
    while (file) {
        Edge edge;
        int32_t s, d, w;
        file.read ((char*)&s, sizeof(int32_t));
        file.read ((char*)&d, sizeof(int32_t));
        file.read ((char*)&w, sizeof(int32_t));
        edge.source = s;
        edge.dest = d;
        edge.weight = w;
        G.edges.push_back(edge);
    }
    file.close();
    return G;
}
