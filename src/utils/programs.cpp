#ifndef PROGRAMS_CPP
#define PROGRAMS_CPP
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

using namespace poplar;
using namespace poplar::program;

std::tuple<Program, Program> buildJaccardReductionProgram(Graph & graph, Tensor & adj, Tensor & out, CSR<vid_t> * g, int num_ipus){
  ComputeSet bst = graph.addComputeSet("bst");	
  ComputeSet reduction = graph.addComputeSet("reduction");	
  unsigned long long int num_bst = 0;
  for(auto src = 0; src < g->num_vertices; src++){
    for(auto dsti = g->V[src]; dsti < g->V[src+1]; dsti++){
      graph.setTileMapping(adj.slice(g->V[src], g->V[src+1]), num_bst % (MK2IPU_TILE_NUM * num_ipus));
      auto dst = g->E[dsti];
      if(src < dst){
        auto reduction_vertex = graph.addVertex(reduction, "Reduce");
        unsigned int i = 0;
        unsigned int s = g->V[dst+1] - g->V[dst]; 
        Tensor reduc = graph.addVariable(BOOL, {s}, "rec"+to_string(dsti));
        graph.setTileMapping(reduc, num_bst % (MK2IPU_TILE_NUM * num_ipus));
        for(auto ptr = g->V[dst]; ptr < g->V[dst+1]; ptr++){
          num_bst++;
          auto bst_vertex = graph.addVertex(bst, "BST");
          graph.connect(bst_vertex["target"], adj.slice(g->V[src], g->V[src+1]));
          graph.connect(bst_vertex["q"], adj[ptr]);
          graph.connect(bst_vertex["res"], reduc[i]);
          graph.setTileMapping(bst_vertex, num_bst % (MK2IPU_TILE_NUM * num_ipus));
          i++;
        }
        graph.connect(reduction_vertex["results"], reduc);
        graph.connect(reduction_vertex["src_neigh_size"], g->V[src+1] - g->V[src]);
        graph.connect(reduction_vertex["dst_neigh_size"], g->V[dst+1] - g->V[dst]);
        graph.connect(reduction_vertex["jac"], out[dsti]);
        graph.setTileMapping(reduction_vertex, num_bst % (MK2IPU_TILE_NUM * num_ipus));
        //std::cout<< "Vertex " << dsti << ":" << std::endl;
        //std::cout<< "Tile " << src % MK2IPU_TILE_NUM << std::endl;
        //std::cout<< "Edge " << src << " " << dst << endl;
        //std::cout<< "Source Neig: ";
        //for(auto i = g->V[src]; i < g->V[src+1]; i++){
        //	std::cout << g->E[i] << " "; 
        //}
        //std::cout << std::endl;
        //std::cout<< "Dest Neig: ";
        //for(auto dst2 = g->V[dst]; dst2 < g->V[dst+1]; dst2++){
        //	std::cout << g->E[dst2] << " "; 
        //}
        //std::cout << std::endl;
        //std::cout<< "--------------------------------------------" << std::endl;
      }
      graph.setTileMapping(out[dsti], num_bst % (MK2IPU_TILE_NUM * num_ipus));
    }
  }
  return std::make_tuple(Execute(bst), Execute(reduction));
}	

Program buildJaccardProgram(Graph & graph, Tensor & adj, Tensor & out, CSR<vid_t> * g, int num_ipus){
  ComputeSet jac = graph.addComputeSet("jac");	
  for(auto src = 0; src < g->num_vertices; src++){
    for(auto dsti = g->V[src]; dsti < g->V[src+1]; dsti++){
      auto dst = g->E[dsti];
      if(src < dst){
        auto v = graph.addVertex(jac, "Jaccard_BST");
        graph.connect(v["src_neigh"], adj.slice(g->V[src], g->V[src+1]));
        graph.connect(v["dst_neigh"], adj.slice(g->V[dst], g->V[dst+1]));
        graph.connect(v["jac"], out[dsti]);
        graph.setTileMapping(v, dsti % (MK2IPU_TILE_NUM * num_ipus));
        //std::cout<< "Vertex " << dsti << ":" << std::endl;
        //std::cout<< "Tile " << src % MK2IPU_TILE_NUM << std::endl;
        //std::cout<< "Edge " << src << " " << dst << endl;
        //std::cout<< "Source Neig: ";
        //for(auto i = g->V[src]; i < g->V[src+1]; i++){
        //	std::cout << g->E[i] << " "; 
        //}
        //std::cout << std::endl;
        //std::cout<< "Dest Neig: ";
        //for(auto dst2 = g->V[dst]; dst2 < g->V[dst+1]; dst2++){
        //	std::cout << g->E[dst2] << " "; 
        //}
        //std::cout << std::endl;
        //std::cout<< "--------------------------------------------" << std::endl;
      }
    }
  }
  return Execute(jac);
}	


Program buildJaccardProgramWithTileMapping(Graph & graph, Tensor & adj, Tensor & out, CSR<vid_t> * g, int num_ipus){
  ComputeSet jac = graph.addComputeSet("jac");	
  unsigned int tile = 0;
  for(auto src = 0; src < g->num_vertices; src++){
    //tile = src;
    graph.setTileMapping(adj.slice(g->V[src], g->V[src+1]), tile % (MK2IPU_TILE_NUM * num_ipus));
    for(auto dsti = g->V[src]; dsti < g->V[src+1]; dsti++){
      tile++;
      auto dst = g->E[dsti];
      if(src < dst){
        auto v = graph.addVertex(jac, "Jaccard_BST");
        graph.connect(v["src_neigh"], adj.slice(g->V[src], g->V[src+1]));
        graph.connect(v["dst_neigh"], adj.slice(g->V[dst], g->V[dst+1]));
        graph.connect(v["jac"], out[dsti]);
        graph.setTileMapping(v, tile % (MK2IPU_TILE_NUM * num_ipus));
        //std::cout<< "Vertex " << dsti << ":" << std::endl;
        //std::cout<< "Tile " << src % MK2IPU_TILE_NUM << std::endl;
        //std::cout<< "Edge " << src << " " << dst << endl;
        //std::cout<< "Source Neig: ";
        //for(auto i = g->V[src]; i < g->V[src+1]; i++){
        //	std::cout << g->E[i] << " "; 
        //}
        //std::cout << std::endl;
        //std::cout<< "Dest Neig: ";
        //for(auto dst2 = g->V[dst]; dst2 < g->V[dst+1]; dst2++){
        //	std::cout << g->E[dst2] << " "; 
        //}
        //std::cout << std::endl;
        //std::cout<< "--------------------------------------------" << std::endl;
      }
      graph.setTileMapping(out[dsti], tile % (MK2IPU_TILE_NUM * num_ipus));
    }
  }
  return Execute(jac);
}
#endif
