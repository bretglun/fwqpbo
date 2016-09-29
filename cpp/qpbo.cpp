#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

#include "Image.h"
#include "QPBO-v1.4.src/QPBO.h"

void makeImage(image<vector<float> >* im,const float* data,int N,int nx,int ny,int nz) {
	vector<int> nstep(N);
	for (int n=0; n<N; n++)	nstep[n] = n*nx*ny*nz;
	for (int i=0; i<nx*ny*nz; i++) {
		im->data[i] = vector<float>(N);
		for (int n=0; n<N; n++) {
			int j = i+nstep[n];
			im->data[i][n]=data[j];
		}
	}
}

void unmakeImage(image<int>* im,int* data,int nx,int ny,int nz) {
	for (int i=0; i<nx*ny*nz; i++) {
		data[i] = im->data[i];
	}
}

void QPBOgc(int nx, int ny, int nz, image<vector<float> >* D, image<vector<float> >* Vx, image<vector<float> >* Vy, image<vector<float> >* Vz, image<int>* label) {
    int numNodes = nx*ny*nz;

	QPBO<float>* MRF = NULL;
    int numEdges = (nx-1)*ny*nz+nx*(ny-1)*nz+nx*ny*(nz-1);
    MRF = new QPBO<float>(numNodes,numEdges);
    MRF->AddNode(numNodes); // add all nodes

    // Add unary terms:
	for (int i=0; i<nx*ny*nz; i++) MRF->AddUnaryTerm(i,D->data[i][0],D->data[i][1]);

    for (int z=0; z<nz; z++) // Add binary terms in x-direction
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx-1; x++) {
                int i = (z*ny+y)*nx+x;
                int j = i+1;
				MRF->AddPairwiseTerm(i,j,imRef(Vx,x,y,z)[0],imRef(Vx,x,y,z)[1],imRef(Vx,x,y,z)[2],imRef(Vx,x,y,z)[3]);
            }
    for (int z=0; z<nz; z++) // Add binary terms in y-direction
		for (int y=0; y<ny-1; y++)
			for (int x=0; x<nx; x++) {
				int i = (z*ny+y)*nx+x;
				int j = i+nx;
				MRF->AddPairwiseTerm(i,j,imRef(Vy,x,y,z)[0],imRef(Vy,x,y,z)[1],imRef(Vy,x,y,z)[2],imRef(Vy,x,y,z)[3]);
			}
    for (int z=0; z<nz-1; z++) // Add binary terms in z-direction
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++) {
                int i = (z*ny+y)*nx+x;
                int j = i+ny*nx;
				MRF->AddPairwiseTerm(i,j,imRef(Vz,x,y,z)[0],imRef(Vz,x,y,z)[1],imRef(Vz,x,y,z)[2],imRef(Vz,x,y,z)[3]);
            }

    MRF->Solve(); // Run QPBO

	for (int i=0; i<numNodes; i++) label->data[i]=MRF->GetLabel(i);
	delete MRF;
}

extern "C" {
__declspec(dllexport) void __cdecl gc(int nx, int ny, int nz, const float* D, const float* Vx, const float* Vy, const float* Vz, int* label)
{
    image<vector<float> >* D_im = new image<vector<float> >(nx,ny,nz,false);
	image<vector<float> >* Vx_im = new image<vector<float> >(nx-1,ny,nz,false);
	image<vector<float> >* Vy_im = new image<vector<float> >(nx,ny-1,nz,false);
	image<vector<float> >* Vz_im = new image<vector<float> >(nx,ny,nz-1,false);

	makeImage(D_im,D,2,nx,ny,nz);
	makeImage(Vx_im,Vx,4,nx-1,ny,nz);
	makeImage(Vy_im,Vy,4,nx,ny-1,nz);
	makeImage(Vz_im,Vz,4,nx,ny,nz-1);

    image<int>* label_im = new image<int>(nx,ny,nz,false);
    QPBOgc(nx, ny, nz, D_im, Vx_im, Vy_im, Vz_im, label_im);

    unmakeImage(label_im,label,nx,ny,nz);

    delete D_im;
    delete Vx_im;
    delete Vy_im;
    delete Vz_im;
	return;
}
}
