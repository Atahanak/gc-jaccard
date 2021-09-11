#include <iostream>
#include <omp.h>
#include <string>

//ipu specific includes
//#include <poplar/DeviceManager.hpp>
//#include <poplar/IPUModel.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

//utilities
#include "utils/device.hpp"
#include "utils/csr.h"
#include "utils/constants.h"
#include "utils/jaccard.cpp"
#include "utils/programs.cpp"

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
  int partition_id = stoi(argv[3]);
  Device device = getDeviceFromOptions(num_ipus, false); //, partition_id);

  Target target = device.getTarget();
  cout << "Using num_ipus = " << num_ipus << ", partition_id = " << partition_id << endl; 

  Graph graph(target);
  graph.addCodelets("src/codelets/bst.cpp");
  //graph.addCodelets("src/codelets/bst.cpp", CodeletFileType::Auto, "-I./src/utils/");

  //dynamic sliced tensors
  //Tensor xadj = popops::createSliceableTensor(graph, VID_T, {g->num_vertices}, {0}, {g->num_vertices/MK2IPU_TILE_NUM}, 0, "xadj");
  //Tensor xadj = graph.addVariable(VID_T, {g->num_vertices}, "xadj");
  //poputil::mapTensorLinearly(graph, xadj, g->num_vertices/MK2IPU_TILE_NUM, 0);
  //graph.setTileMapping(xadj, 0);
  //Tensor adj = popops::createSliceableTensor(graph, VID_T, {g->num_edges}, {0}, {g->num_vertices/MK2IPU_TILE_NUM}, 0, "adj");
  Tensor adj = graph.addVariable(VID_T, {g->num_edges}, "adj");
  //poputil::mapTensorLinearly(graph, adj, g->num_edges/(MK2IPU_TILE_NUM*num_ipus), 1);
  //graph.setTileMapping(adj, 0);
  //Tensor out = popops::createSliceableTensor(graph, JAC_T, {g->num_edges}, {0}, {g->num_vertices/MK2IPU_TILE_NUM}, 0, "out");
  Tensor out = graph.addVariable(JAC_T, {g->num_edges}, "out");
  //poputil::mapTensorLinearly(graph, out, g->num_edges/(MK2IPU_TILE_NUM*num_ipus), 1);
  //graph.setTileMapping(out, 0);

  //DataStream inStreamXadj = graph.addHostToDeviceFIFO("xadj", VID_T, g->num_vertices); 
  DataStream inStreamAdj = graph.addHostToDeviceFIFO("adj", VID_T, g->num_edges); 
  DataStream outStream = graph.addDeviceToHostFIFO("out", JAC_T, g->num_edges);

  double start = omp_get_wtime();
#ifdef BST_PER_THREAD
  auto programs = buildJaccardReductionProgram(graph, adj, out, g, num_ipus);
#else
  auto jacprog = buildJaccardProgramWithTileMapping(graph, adj, out, g, num_ipus);
#endif
  double end = omp_get_wtime();
  cout << "Graph construction took " << end - start << " seconds." << endl;
  auto input_copy = Sequence(Copy(inStreamAdj, adj));
  auto output_copy = Sequence(Copy(out, outStream));
  start = omp_get_wtime();
#ifdef BST_PER_THREAD
  Engine engine(graph, {input_copy, std::get<0>(programs), std::get<1>(programs), output_copy});
#else
  Engine engine(graph, {input_copy, jacprog, output_copy});
#endif
  end = omp_get_wtime();
  cout << "Graph compilation took " << end - start << " seconds." << endl;
  engine.load(device);
  //engine.connectStream("xadj", g->V);
  engine.connectStream("adj", g->E);
  engine.connectStream("out", output);

  start = omp_get_wtime();
  engine.run(0);
  end = omp_get_wtime();
  std::cout << "Input copy completed in " << end - start << " seconds." << endl;
  start = omp_get_wtime();
  engine.run(1);
  end = omp_get_wtime();
  std::cout << "Jaccard completed in " << end - start << " seconds." << endl;
#ifdef BST_PER_THREAD
  std::cout << "Reducing bst results...\n";
  start = omp_get_wtime();
  engine.run(2);
  end = omp_get_wtime();
  std::cout << "Reduction completed in " << end - start << " seconds." << endl;
#endif
  start = omp_get_wtime();
#ifdef BST_PER_THREAD
  engine.run(3);
#else
  engine.run(2);
#endif
  end = omp_get_wtime();
  std::cout << "Output copy completed in " << end - start << " seconds." << endl;

  jac_t * jaccards = new jac_t[g->num_edges]();
  start = omp_get_wtime();
  jaccard_cpu_markers(g, jaccards); 
  end = omp_get_wtime();
  std::cout << "CPU alg completed in " << end - start << " seconds." << endl;
  cout << "Correct: " << correct(output, jaccards, g->num_edges) << std::endl;
  return 0;

}
