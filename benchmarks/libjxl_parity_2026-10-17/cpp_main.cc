#include "butteraugli/butteraugli.h"
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
std::vector<butteraugli::ImageF> read(const char* path,int w,int h){
 std::ifstream f(path,std::ios::binary);std::vector<unsigned char> b(w*h*3); f.read((char*)b.data(),b.size()); if(!f)std::abort();
 std::vector<butteraugli::ImageF> v;for(int c=0;c<3;c++)v.emplace_back(w,h);
 for(int y=0;y<h;y++)for(int x=0;x<w;x++)for(int c=0;c<3;c++) {double s=b[(y*w+x)*3+c];v[c].Row(y)[x]=s<=10.31475?s/12.92:255.*std::pow((s/255.+.055)/1.055,2.4);}
 return v;
}
int main(int argc,char**argv){int w=atoi(argv[1]),h=atoi(argv[2]);auto r=read(argv[3],w,h),d=read(argv[4],w,h);butteraugli::ImageF dm;double score;if(!butteraugli::ButteraugliInterface(r,d,1.0,dm,score))return 2;printf("%.17g\n",score);}
