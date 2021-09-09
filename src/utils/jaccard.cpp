#ifndef JACCARD_CPP
#define JACCARD_CPP
#include <omp.h>
#include"constants.h"

void jaccard_cpu_markers(CSR<vid_t>* g, jac_t *jaccards)
{
#pragma omp parallel 
  {
    long long int *markers = new long long int[g->num_vertices];
    for(long long int i = 0; i < g->num_vertices; i++){
      markers[i] = (long long int)-1;
    }
#pragma omp for schedule(dynamic, 256)
    for (long long int u = 0; u < g->num_vertices; u++)
    {
      for (long long int ptr = g->V[u]; ptr < g->V[u + 1]; ptr++)
      {
        markers[g->E[ptr]] = u;
      }

      for (unsigned long long int ptr = g->V[u]; ptr < g->V[u + 1]; ptr++)
      {
        long long int v = g->E[ptr];
        if(u < v){
          int intersection_size = 0;
          for(long long int ptr_v = g->V[v]; ptr_v < g->V[v + 1]; ptr_v++){
            long long int w = g->E[ptr_v];
            if(w!=u && markers[w] == u){
              long long int sizew = g->V[w + 1] - g->V[w];
              intersection_size++;
            }
            jaccards[((unsigned long long int)ptr)] = 1.0*(intersection_size)/((g->V[u+1]-g->V[u])+(g->V[v+1]-g->V[v])-intersection_size);
          }
        }
      }
    }
    delete[] markers;
  }
}
#endif
