#include <iostream>
#include <omp.h>
#include <string>

//ipu specific includes
//#include <poplar/DeviceManager.hpp>
//#include <poplar/IPUModel.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

//utilities
#include "utils/device.hpp"
#include "utils/csr.h"
#include "utils/constants.h"
#include "utils/jaccard.cpp"

using namespace poplar;
using namespace poplar::program;

void usage(){
  std::cout<<"Input Error!" << std::endl;
  std::cout<<"Usage: <bcsr_path> <num_ipus>" << std::endl;
}

void printDeviceAttributes(Device & device){
  std::map<std::string, std::string> device_attrbs = getDeviceAttributes(device);
  for (const auto &attr : device_attrbs) {
    std::cout<< attr.first << " " << attr.second << std::endl;
  }
}

bool correct(jac_t * out, jac_t * jaccards, vid_t size){
  //std::cout<< "Results:" << std::endl;
  for(vid_t i = 0; i < size; i++){
    //std::cout << out[i] << std::endl;
    if(out[i] != jaccards[i]){
      return false;
    }
  }
  return true;
}

Program buildJaccardProgram(Graph & graph, Tensor & xadj, Tensor & adj, Tensor & out, CSR<vid_t> * g, int num_ipus){
  ComputeSet jac = graph.addComputeSet("jac");	
  for(auto src = 0; src < g->num_vertices; src++){
    //if(src+1 > MK2IPU_TILE_NUM){
    //	break;
    //}
    for(auto dsti = g->V[src]; dsti < g->V[src+1]; dsti++){
      auto dst = g->E[dsti];
      if(src < dst){
        auto v = graph.addVertex(jac, "Jaccard_BST");
        graph.connect(v["src_neigh"], adj.slice(g->V[src], g->V[src+1]));
        graph.connect(v["dst_neigh"], adj.slice(g->V[dst], g->V[dst+1]));
        graph.connect(v["jac"], out[dsti]);
        graph.setTileMapping(v, src % (MK2IPU_TILE_NUM * num_ipus));
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

//bool binarySearch(std::vector<vid_t> & target, const unsigned int & q){
//  int start = 0;
//  int end = target.size() - 1;
//  int middle;
//  while(start <= end){
//    middle = start + (end - start)/2;
//    if(target[middle] == q){
//      return true;
//    }
//    else if(target[middle] < q){
//      start = middle + 1;
//    }
//    else{
//      end = middle - 1;
//    }	
//  }
//  return false;
//}
//
//void testBinarySearch(CSR<vid_t> * g){
//  for(auto src = 0; src < g->num_vertices; src++){
//    for(auto dsti = g->V[src]; dsti < g->V[src+1]; dsti++){
//      auto dst = g->E[dsti];
//      printf("src : %u, dest: %u\n", src, dst);
//      for(auto neig = g->V[dst]; neig < g->V[dst+1]; neig++){
//        std::vector<vid_t> vec(g->E + g->V[src], g->E + g->V[src+1]);
//        std::cout << "neig: " << g->E[neig] << endl;
//        for(int i =0; i < vec.size(); i++){
//          std::cout<< vec[i] << std::endl;
//        }
//        binarySearch(vec, g->E[neig]);
//      }
//    }
//  }
//}

int main(int argc, char * argv[]){
  if(argc < 3){
    usage();
    return 0;
  }
  //
  cerr << MAX_TILE_MEMORY_SIZE << " bytes are allowed per tile." << endl;
  //read graph
  CSR<vid_t> * g = new CSR<vid_t>(argv[1]);
  std::cout<< argv[1] << "=> #vertices = " << g->num_vertices << " #edges = " << g->num_edges << endl;
  jac_t * output = new jac_t[g->num_edges];
  
  //get ipu/(s)
  int num_ipus = stoi(argv[2]);
  Device device = getDeviceFromOptions(num_ipus, false);

  Target target = device.getTarget();

  Graph graph(target);
  graph.addCodelets("src/codelets/bst.cpp");
  //graph.addCodelets("src/codelets/bst.cpp", CodeletFileType::Auto, "-I./src/utils/");

  Tensor xadj = graph.addVariable(VID_T, {g->num_vertices}, "xadj");
  graph.setTileMapping(xadj, 0);
  Tensor adj = graph.addVariable(VID_T, {g->num_edges}, "adj");
  graph.setTileMapping(adj, 0);
  Tensor out = graph.addVariable(JAC_T, {g->num_edges}, "out");
  graph.setTileMapping(out, 0);

  DataStream inStreamXadj = graph.addHostToDeviceFIFO("xadj", VID_T, g->num_vertices); 
  DataStream inStreamAdj = graph.addHostToDeviceFIFO("adj", VID_T, g->num_edges); 
  DataStream outStream = graph.addDeviceToHostFIFO("out", JAC_T, g->num_edges);

  double start = omp_get_wtime();
  auto jacprog = buildJaccardProgram(graph, xadj, adj, out, g, num_ipus);
  double end = omp_get_wtime();
  cout << "Graph construction took " << end - start << " seconds." << endl;
  auto input_copy = Sequence(Copy(inStreamXadj, xadj), Copy(inStreamAdj, adj));
  auto output_copy = Sequence(Copy(out, outStream));
  start = omp_get_wtime();
  Engine engine(graph, {input_copy, jacprog, output_copy});
  end = omp_get_wtime();
  cout << "Graph compilation took " << end - start << " seconds." << endl;
  engine.load(device);
  engine.connectStream("xadj", g->V);
  engine.connectStream("adj", g->E);
  engine.connectStream("out", output);

  std::cout << "Copying inputs to the device...\n";
  start = omp_get_wtime();
  engine.run(0);
  end = omp_get_wtime();
  std::cout << "Input copy completed in " << end - start << " seconds." << endl;
  std::cout << "Running jaccard on the device...\n";
  start = omp_get_wtime();
  engine.run(1);
  end = omp_get_wtime();
  std::cout << "Jaccard completed in " << end - start << " seconds." << endl;
  std::cout << "Copying outputs to the host...\n";
  start = omp_get_wtime();
  engine.run(2);
  end = omp_get_wtime();
  std::cout << "Output copy completed in " << end - start << " seconds." << endl;

  jac_t * jaccards = new jac_t[g->num_edges]();
  std::cout << "Checking correctness...\n";
  start = omp_get_wtime();
  engine.run(2);
  end = omp_get_wtime();
  std::cout << "Check completed in " << end - start << " seconds." << endl;
  jaccard_cpu_markers(g, jaccards); 
  cout << "Correct: " << correct(output, jaccards, g->num_edges);
  //printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  //Tensor src_neigh = graph.addVariable(UNSIGNED_INT, {8}, "src_neigh");
  //Tensor queries = graph.addVariable(UNSIGNED_INT, {3}, "queries");
  //Tensor count = graph.addVariable(UNSIGNED_INT, {1}, "count");
  //
  //// Create a control program that is a sequence of steps
  //Sequence prog;

  //graph.setTileMapping(src_neigh, 1);
  //graph.setTileMapping(queries, 1);
  //graph.setTileMapping(count, 1);
  //Tensor c1 = graph.addConstant<unsigned int>(UNSIGNED_INT, {8}, {1, 5, 7, 10, 13, 15, 18, 21});
  //graph.setTileMapping(c1, 0);
  //prog.add(Copy(c1, src_neigh));
  //Tensor c2 = graph.addConstant<unsigned int>(UNSIGNED_INT, {3}, {1, 5, 3});
  //graph.setTileMapping(c2, 0);
  //prog.add(Copy(c2, queries));

  //std::cout<< "Copies are added to the program." << std::endl;
  //
  //ComputeSet computeSet = graph.addComputeSet("computeSet");
  //VertexRef vtx = graph.addVertex(computeSet, "CN_BST");
  //graph.connect(vtx["target"], src_neigh.slice(0, 8));
  //graph.connect(vtx["queries"], queries.slice(0, 3));
  //graph.connect(vtx["count"], count[0]);
  //graph.setTileMapping(vtx, 1);
  //graph.setPerfEstimate(vtx, 20);
  //prog.add(Execute(computeSet));
  //std::cout<< "Bst is added to the program." << std::endl;


  //prog.add(PrintTensor("count", count));

  //Engine engine(graph, prog);
  //engine.load(device);

  //// Run the control program
  //std::cout << "Running program\n";
  //engine.run(0);
  //std::cout << "Program complete\n";

  return 0;

}
