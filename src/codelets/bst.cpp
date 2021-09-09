#include <poplar/Vertex.hpp>
//#include "./constants.h"

typedef unsigned int vid_t;
typedef float jac_t;

//returns the jaccard value between two vertices 
class Jaccard_BST : public poplar::Vertex {
	public:
		poplar::Input<poplar::Vector<vid_t>> src_neigh;
		poplar::Input<poplar::Vector<vid_t>> dst_neigh;
		poplar::Output<jac_t> jac;

		unsigned int binarySearch(poplar::Vector<vid_t> & target, const vid_t & q){
			int start = 0;
			int end = target.size() - 1;
			int middle;
			while(start <= end){
				middle = start + (end - start)/2;
				if(target[middle] == q){
					return 1;
				}
				else if(target[middle] < q){
					start = middle + 1;
				}
				else{
					end = middle - 1;
				}	
			}
			return 0;
		}
		
		bool compute(){
			unsigned int cn = 0;
			for (const auto &v : dst_neigh) {
			      cn += binarySearch(src_neigh, v);
			}
			*jac = (float)cn / (src_neigh.size() + dst_neigh.size() - cn); 
			return true;
		}
};


class BST : public poplar::Vertex {
	public:
		poplar::Input<poplar::Vector<vid_t>> target;
		poplar::Input<vid_t> q;
		poplar::Output<bool> res;

		unsigned int binarySearch(poplar::Vector<vid_t> & target, const vid_t & q){
			int start = 0;
			int end = target.size() - 1;
			int middle;
			while(start <= end){
				middle = start + (end - start)/2;
				if(target[middle] == q){
					return 1;
				}
				else if(target[middle] < q){
					start = middle + 1;
				}
				else{
					end = middle - 1;
				}	
			} 
			return 0;
		}
		
		bool compute(){
      *res = binarySearch(target, *q);
			return true;
		}
};

class Reduce : public poplar::Vertex {
  public:
    poplar::Input<poplar::Vector<vid_t>> results;
    poplar::Input<vid_t> src_neigh_size;
    poplar::Input<vid_t> dst_neigh_size;
    poplar::Output<jac_t> jac;

    bool compute(){
      vid_t cn = 0; 
      for(const auto &r : results){
        cn += (vid_t)r;
      }
			*jac = (float)cn / (src_neigh_size + dst_neigh_size - cn); 
    }
};
