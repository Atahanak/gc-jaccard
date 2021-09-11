#ifndef _DEVICE
#define _DEVICE


#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <utility>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>

inline
poplar::Device getIpuModelDevice(std::size_t numIpus) {
  poplar::IPUModel ipuModel("ipu2");
  ipuModel.numIPUs = numIpus;
  return ipuModel.createDevice();
}

size_t total_used_memory(std::vector<std::tuple<std::string, std::pair<size_t, size_t>, poplar::Graph::TileToTensorMapping>> & mc){
   size_t used_memory = 0;
   for (auto & t : mc){
        used_memory+=std::get<1>(t).first;
   }
   return used_memory;
}

inline std::pair<size_t, size_t> size_of_tensor(poplar::Tensor t, const poplar::Target & target){
    return std::make_pair(t.numElements() * target.getTypeSize(t.elementType()), target.getTypeSize(t.elementType()));
}
inline std::vector<size_t> memory_per_tile(std::vector<std::tuple<std::string, std::pair<size_t, size_t>, poplar::Graph::TileToTensorMapping>> & mc){
  std::vector<size_t> size(1500,0);
    for (auto & t : mc){
        auto e_size = std::get<1>(t).second;
        for (int i =0; i<std::get<2>(t).size(); i++){
          auto tile = std::get<2>(t)[i];
          for (auto & interval : tile){
            size[i]+= interval.size()*e_size;
          }
        }
    }
    return size;
}
// Return a HW device with the requested number of IPUs.
// Exception is thrown if no devices with the requested
// number are available.
inline
poplar::Device getIpuHwDevice(std::size_t numIpus) {
  auto dm = poplar::DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
  std::cerr << "Number of devices: " << hwDevices.size() << std::endl;
  if (hwDevices.size() > 0) {
    for (auto &d : hwDevices) {
      if (d.attach()) {
        return std::move(d);
      } else {
        std::string error = "IPU id " + std::to_string(d.getId()) + " failed to attach.";
        //throw std::runtime_error(error);
      }
    }
  }
  throw std::runtime_error("No IPU hardware available.");
}

inline
poplar::Device getIpuHwDeviceById(std::size_t id) {
  auto dm = poplar::DeviceManager::createDeviceManager();
  auto hwDevice = dm.getDevice(id);
  if (hwDevice.attach()) {
    return std::move(hwDevice);
  } else {
    std::string error = "IPU id " + std::to_string(hwDevice.getId()) + " failed to attach.";
    throw std::runtime_error(error);
  }
}

inline 
void printTensorMapping(const poplar::Graph& graph, const poplar::Tensor & tensor, const char* name="_"){
  auto mapping =  graph.getTileMapping(tensor);
  std::cout << name << " mapping: "<< std::endl;
  for (int i =0; i<mapping.size(); i++){
    if (mapping[i].size()>0){
      std::cout << "Tile " << i << ": ";
      for (int j = 0; j < mapping[i].size(); j++){
        std::cout << mapping[i][j] << ",\t"; 
      }
      std::cout << std::endl;
    }
  }
}
 
inline
poplar::Device getDeviceFromOptions(int num_ipus, const bool use_IPU_model, const int id = -1) {
  poplar::Device device;
  if(use_IPU_model == true) {
    device = getIpuModelDevice(num_ipus);
    std::cerr << "Using IPU model\n";
  } else {
    if(id == -1){
      device = getIpuHwDevice(num_ipus);
    }
    else{
      device = getIpuHwDeviceById(id);
    }
    std::cerr << "Using HW device ID: " << device.getId() << "\n";
  }
  return device;
}

inline
std::map<std::string, std::string> getDeviceAttributes(poplar::Device & device){
	return device.getAttributes();
}
#endif
