#include <ctime>
#include <iostream>
#include <complex>
#include <stdexcept>
using namespace std;

#include <Eigen/Dense>

using Eigen::MatrixXcf;
using Eigen::VectorXcf;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;

#include "Image.h"
#include "QPBO-v1.4.src/QPBO.h"
#include "fibonacci.h"

#define PI 3.14159265358979323846
#define gyro 42.576 // 1H Gyromagnetic ratio [MHz/T]
#define IMGTYPE float
#define RESIDUALTYPE float // Expects floating point type
#define MRFTYPE float // Can be int, float or double

template <class T>
inline T square(const T &x) { return x*x; };

MatrixXcf modelMatrix(float t1, float dt, float B0, int N, float* CS, float* alpha, int M, int P) {
	MatrixXcf A = MatrixXcf(N,M);
	for (int n=0; n<N; n++) { // Loop over echoes
		float t = t1+n*dt;
		for (int m=0; m<M; m++) { // Loop over species
            A(n,m) = complex<float>(0,0);
			for (int p=0; p<P; p++) { // Loop over all resonances
				float omega = 2.0*PI*gyro*B0*(CS[p]-CS[0]); // Chemical shift between peak p and peak 0 (ppm)
				A(n,m) += alpha[p+m*P]*exp(complex<float>(0,t*omega));
			}
		}
    }
	return A;
}

vector<MatrixXcf> modulationMatrices(int nB0, int N) { // Calculate off-resonance phase modulation vectors
	vector<MatrixXcf> B(nB0);
	for (int b=0; b<nB0; b++) {
		float omega = 2.0*PI*b/nB0;
		B[b] = MatrixXcf::Identity(N,N);
		for(int n=0;n<N;n++) B[b](n,n) = exp(complex<float>(0,n*omega));
	}
	return B;
}

vector<MatrixXcf> decayMatrices(int nR2, float R2step, int N, float t1, float dt) { // Calculate R2* decay vectors
	vector<MatrixXcf> R(nR2);
	for (int r=0; r<nR2; r++) {
		float R2 = r*R2step;
		R[r] = MatrixXcf::Identity(N,N);
		for(int n=0;n<N;n++) {
            //float t = t1+n*dt;
            float t = n*dt; // Include e^(-t1*R2) in W and F to reduce noise. Does not affect fat fraction
            R[r](n,n) = exp(complex<float>(-t*R2,0));
        }
	}
	return R;
}

MatrixXcf pseudoinverse(MatrixXcf A) {
	MatrixXcf AtA = A.adjoint()*A;
	MatrixXcf Ap=AtA.inverse()*A.adjoint();
	return Ap;
}

MatrixXcf nullspaceProjectionMatrix(MatrixXcf A, MatrixXcf Ap, int N) {
	MatrixXcf eye = MatrixXcf::Identity(N,N);
	return eye-A*Ap;
}

void createEchoImages(image<VectorXcf>* S,const IMGTYPE* re,const IMGTYPE* im,int N,int nx,int ny,int nz) {
	vector<int> nstep(N);
	for (int n=0; n<N; n++)	nstep[n] = n*nx*ny*nz;
	for (int i=0; i<nx*ny*nz; i++) {
		S->data[i] = VectorXcf(N);
		for (int n=0; n<N; n++) {
			int j = i+nstep[n];
			S->data[i](n)=complex<float>(re[j],im[j]);
		}
	}
}

RESIDUALTYPE residual(VectorXcf &Cvec, VectorXcf &Svec) {
    RESIDUALTYPE res = 0;
    for (int k=0; k<Cvec.size(); k++) res+=real(Cvec(k)*Svec(k));
    return res;
}

// Progressbar: progress is i out of n
static inline void progressBar(int i, int n)
{
	if (i%((n-1)/101)!= 0) return; // Avoid printing to cout unless updated
	cout << "\r" << (int)(i*100/(n-1)) << "% complete...";
	cout.flush();
}

void getResidualImages(image<vector<RESIDUALTYPE> >* J,image<VectorXcf>* S,vector<vector<VectorXcf> > Cvec,vector<vector<int> > lowTriInd,int N,int nB0,int* iR2cand,int nR2cand,int nVxl) {
    vector<RESIDUALTYPE> res(nB0);
    VectorXcf Svec = VectorXcf(Cvec.size());
    for (int i=0; i<nVxl; i++) {
        progressBar(i,nVxl);
        VectorXcf s = S->data[i];
        for (int m=0; m<N; m++)
            for (int n=0; n<=m; n++)
                Svec(lowTriInd[m][n]) = conj(s(m))*s(n);
        // Calculate residual for each label
        for (int b=0; b<nB0; b++) {
            res[b] = residual(Cvec[b][iR2cand[0]],Svec);
            for (int r=1; r<nR2cand; r++) { // Loop over R2* candidates
                RESIDUALTYPE val = residual(Cvec[b][iR2cand[r]],Svec);
                if (val<res[b]) res[b] = val;
            }
        }
        J->data[i] = res;
    }
}

// Find global minimum of discretely evaluated function f(t) with period T. If multiple equal minima, the one closest to 0 or T is returned.
pair<RESIDUALTYPE,int> findGlobalMinimum(vector<RESIDUALTYPE> f, int T) {
	pair<RESIDUALTYPE,int> minimum = pair<RESIDUALTYPE,int>(f[0], 0);
	for (int t=1; t<T; t++) {
		int index = t/2+1;
		if (t%2==0) index = T-t/2;
		if (f[index]<minimum.first)	minimum = pair<RESIDUALTYPE,int>(f[index], index);
	}
	return minimum;
}

// Find all local minima of discretely evaluated function f(t) with period T
vector<pair<RESIDUALTYPE,int> > findLocalMinima(vector<RESIDUALTYPE> f, int T) {
	vector<pair<RESIDUALTYPE,int> > minima;
	RESIDUALTYPE prev = f[T-1];
	RESIDUALTYPE cur;
	bool falling = false;
	for (int t=0; t<T+1; t++) {
		cur = f[t%T];
		if (cur<prev) falling=true; // cur is a minimum candidate
		else if (falling) { // prev was a minimum
			minima.push_back(pair<RESIDUALTYPE,int>(prev,(t-1)%T));
			falling=false;
		}
		prev=cur;
	}
	return minima;
}

// In each voxel, find the two smallest local residual minima in a period of omega
void findTwoSsmallestMinima(image<vector<RESIDUALTYPE> >* J, image<int>* min1, image<int>* min2, int nB0) {
	for (int i=0; i<J->xsize()*J->ysize()*J->zsize(); i++) {
		vector<pair<RESIDUALTYPE,int> > minima = findLocalMinima(J->data[i],nB0);
		if (minima.size()>1) {
			sort(minima.begin(), minima.end());
			int m1 = minima[0].second;
			min1->data[i] = m1;
			int m2 = minima[1].second;
			min2->data[i] = m2;
		} else if (minima.size()==1) {
			int m1 = minima[0].second;
			min1->data[i] = m1;
			min2->data[i] = m1;
		} else {
			min1->data[i] = 0; // Assign dummy minimum
			min2->data[i] = 0; // Assign dummy minimum
		}
	}
}

// Exhaustive (brute force) search in interval [0,nR2-1]
unsigned int ExhaustiveSearch(VectorXcf &s, vector<VectorXcf> Cvec, int nR2, int N, vector<vector<int> > lowTriInd) {
    VectorXcf Svec = VectorXcf(Cvec.size());
    for (int m=0; m<N; m++)
        for (int n=0; n<=m; n++)
            Svec(lowTriInd[m][n]) = conj(s(m))*s(n);

	unsigned int best = 0;
	RESIDUALTYPE value;
	RESIDUALTYPE min_value = residual(Cvec[0],Svec);
	for (int r=1; r<nR2; r++) {
		value = residual(Cvec[r],Svec);
		if (value<min_value) {
			min_value = value;
			best = r;
		}
	}
	return best;
}

// Fibonacci search in interval [0,N-1]. Expects N>2
unsigned int FibonacciSearch(VectorXcf &s, vector<VectorXcf> Cvec, int nR2, int N, vector<vector<int> > lowTriInd) {
    VectorXcf Svec = VectorXcf(Cvec.size());
    for (int m=0; m<N; m++)
        for (int n=0; n<=m; n++)
            Svec(lowTriInd[m][n]) = conj(s(m))*s(n);

	unsigned int Fib_index;
	unsigned int* Fib = get_Fibonacci_sequence_with_final_number(nR2-1, Fib_index);
	unsigned int i = Fib_index-1;
	unsigned int a = 0;
	unsigned int x1 = Fib[i-1];
	unsigned int x2 = Fib[i];
	unsigned int b = Fib[i+1];
	RESIDUALTYPE fa = residual(Cvec[a],Svec);
	RESIDUALTYPE fx1 = residual(Cvec[x1],Svec);
	RESIDUALTYPE fx2 = residual(Cvec[x2],Svec);
	RESIDUALTYPE fb = residual(Cvec[b],Svec);
	while ( i > 0 )
		if (fx1 > fx2) {
			a = x1;
			fa = fx1;
			x1 = x2;
			fx1 = fx2;
			if ( --i < 2 ) {
				if (fa<=fx1) return a;
				else if (fx1<=fb) return x1;
				else return b;
			}
			x2 = a + Fib[i];
			fx2 = residual(Cvec[x2],Svec);
		} else {
			b = x2;
			fb = fx2;
			x2 = x1;
			fx2 = fx1;
			if ( --i < 2 ) {
				if (fa<=fx2) return a;
				else if (fx2<=fb) return x2;
				else return b;
			}
			x1 = a + Fib[i - 1];
			fx1 = residual(Cvec[x1],Svec);
		}
	return 0;
}

// find J'' at each local minimum by second order finite differences
void find2ndDerivativeAtMinimum(image<vector<RESIDUALTYPE> >* J,image<RESIDUALTYPE>* ddJ,image<int>* mini,int nB0,int nx,int ny,int nz) {
	for (int i=0; i<nx*ny*nz; i++) {
		vector<RESIDUALTYPE> res = J->data[i];
		int b = mini->data[i];
		// NOTE: No division by square(steplength) since square(steplength) not included in V
		ddJ->data[i] = res[(b+nB0-1)%nB0]+res[(b+1)%nB0]-2.0*res[b];
	}
}

void calcWeights(image<float>* wx,image<float>* wy,image<float>* wz,image<RESIDUALTYPE>* ddJ,float mu,float dx,float dy,float dz,int nx,int ny,int nz) {
	float commonX = mu/dx;
	float commonY = mu/dy;
	float commonZ = mu/dz;
	for (int z=0; z<nz; z++)
		for (int y=0; y<ny; y++)
			for (int x=0; x<nx-1; x++) imRef(wx,x,y,z) = commonX*min(imRef(ddJ,x,y,z),imRef(ddJ,x+1,y,z));
	for (int z=0; z<nz; z++)
		for (int y=0; y<ny-1; y++)
			for (int x=0; x<nx; x++) imRef(wy,x,y,z) = commonY*min(imRef(ddJ,x,y,z),imRef(ddJ,x,y+1,z));
	for (int z=0; z<nz-1; z++)
		for (int y=0; y<ny; y++)
			for (int x=0; x<nx; x++) imRef(wz,x,y,z) = commonZ*min(imRef(ddJ,x,y,z),imRef(ddJ,x,y,z+1));
}

void addTermsToMRF(QPBO<MRFTYPE>* q,image<vector<RESIDUALTYPE> >* J,image<int>* min1,image<int>* min2,image<float>* wx,image<float>* wy,image<float>* wz,vector<float> V,int nx,int ny,int nz) {
    // Add unary terms:
	for (int i=0; i<nx*ny*nz; i++) q->AddUnaryTerm(i,J->data[i][min1->data[i]],J->data[i][min2->data[i]]);

    for (int z=0; z<nz; z++) // Add binary terms in x-direction
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx-1; x++) {
            	float w = imRef(wx,x,y,z);
                int i = (z*ny+y)*nx+x;
                int j = i+1;
				q->AddPairwiseTerm(i,j,w*V[abs(imRef(min1,x,y,z)-imRef(min1,x+1,y,z))],w*V[abs(imRef(min1,x,y,z)-imRef(min2,x+1,y,z))],w*V[abs(imRef(min2,x,y,z)-imRef(min1,x+1,y,z))],w*V[abs(imRef(min2,x,y,z)-imRef(min2,x+1,y,z))]);
            }
    for (int z=0; z<nz; z++) // Add binary terms in y-direction
		for (int y=0; y<ny-1; y++)
			for (int x=0; x<nx; x++) {
				float w = imRef(wy,x,y,z);
				int i = (z*ny+y)*nx+x;
				int j = i+nx;
				q->AddPairwiseTerm(i,j,w*V[abs(imRef(min1,x,y,z)-imRef(min1,x,y+1,z))],w*V[abs(imRef(min1,x,y,z)-imRef(min2,x,y+1,z))], w*V[abs(imRef(min2,x,y,z)-imRef(min1,x,y+1,z))], w*V[abs(imRef(min2,x,y,z)-imRef(min2,x,y+1,z))]);
			}
    for (int z=0; z<nz-1; z++) // Add binary terms in z-direction
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++) {
            	float w = imRef(wz,x,y,z);
                int i = (z*ny+y)*nx+x;
                int j = i+ny*nx;
				q->AddPairwiseTerm(i,j,w*V[abs(imRef(min1,x,y,z)-imRef(min1,x,y,z+1))], w*V[abs(imRef(min1,x,y,z)-imRef(min2,x,y,z+1))], w*V[abs(imRef(min2,x,y,z)-imRef(min1,x,y,z+1))], w*V[abs(imRef(min2,x,y,z)-imRef(min2,x,y,z+1))]);
            }
}

// Define a struct to hold each point.
struct Neighbor {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    float weight;
};

// Returns neighbors in 6-neighborhood
vector<Neighbor> getNeighbors(int x,int y,int z,int nx,int ny,int nz,image<float>* wx,image<float>* wy,image<float>* wz) {
	vector<Neighbor> ngbs;
	Neighbor ngb;
	if (x+1<nx) { // Right
		ngb.x=x+1; ngb.y=y; ngb.z=z; ngb.weight=imRef(wx,x,y,z);
		ngbs.push_back(ngb);
	} if (x>0) { // Left
		ngb.x=x-1; ngb.y=y; ngb.z=z; ngb.weight=imRef(wx,x-1,y,z);
		ngbs.push_back(ngb);
	} if (y+1<ny) { // Up
		ngb.x=x; ngb.y=y+1; ngb.z=z; ngb.weight=imRef(wy,x,y,z);
		ngbs.push_back(ngb);
	} if (y>0) { // Down
		ngb.x=x; ngb.y=y-1; ngb.z=z; ngb.weight=imRef(wy,x,y-1,z);
		ngbs.push_back(ngb);
	} if (z+1<nz) { // Above
		ngb.x=x; ngb.y=y; ngb.z=z+1; ngb.weight=imRef(wz,x,y,z);
		ngbs.push_back(ngb);
	}if (z>0) { // Below
		ngb.x=x; ngb.y=y; ngb.z=z-1; ngb.weight=imRef(wz,x,y,z-1);
		ngbs.push_back(ngb);
	}
	return ngbs;
}

// Precalculate label traverse orders
vector<vector<int> > getOrders(int L) {
	vector<int> basicOrder(L); // Basic labels traverse order
	for (int o=0; o<L; o++) o%2==0 ? basicOrder[o] = L+o/2 : basicOrder[o] = L-1-o/2; // even or odd?
	vector<vector<int> > orders(L, vector<int>(L));
	for (int l=0; l<L; l++)	for (int o=0; o<L; o++) orders[l][o]=(basicOrder[o]+l)%L; // Offset basic order about label l
	return orders;
}

void ICM(image<int>* current,int L,int maxLabelUpdate,int nIter,image<vector<RESIDUALTYPE> >* J,vector<float> V,image<float>* wx,image<float>* wy,image<float>* wz,int nx,int ny,int nz) {
	image<int>* previous = new image<int>(nx,ny,nz,false);
	vector<vector<int> > orders = getOrders(L); // Precalculate label traverse orders
	int max_o = min(L,1+maxLabelUpdate*2);
	for (int k=0; k<nIter; k++) { // ICM iterate
		cout << k+1 << ", ";
		for (int i=0; i<nx*ny*nz; i++)	previous->data[i] = current->data[i]; // Update prev image
		// Then update labels:
		for (int z=0; z<nz; z++)
			for (int y=0; y<ny; y++)
				for (int x=0; x<nx; x++) {
					vector<Neighbor> ngbs = getNeighbors(x,y,z,nx,ny,nz,wx,wy,wz);
					int prev = imRef(previous,x,y,z);
					vector<int> order = orders[prev]; // Centric order about previous label
					int best = prev;
					MRFTYPE min_cost = std::numeric_limits<MRFTYPE>::max();
					for (int o=0; o<max_o; o++) { // For all labels (in order order)
						int l = order[o];
						MRFTYPE cost = imRef(J,x,y,z)[l]; // Unary cost
						for (unsigned int n = 0; cost<min_cost && n<ngbs.size(); n++) { // Loop over neighbors, abort if cost exceeds min_cost
							int ngb_label = imRef(previous,ngbs[n].x,ngbs[n].y,ngbs[n].z);
							cost += ngbs[n].weight*V[abs(l-ngb_label)]; // Add binary cost (weights already scaled)
						}
						if (cost<min_cost) {
							min_cost = cost;
							best = l;
						}
					}
					imRef(current,x,y,z) = best;
				}
	}
	delete previous;
 }

struct Level {
    int L; // Level index
    int nx; // Number of voxels in x dir
    int ny; // Number of voxels in y dir
    int nz; // Number of voxels in z dir
	int sx; // Number of child voxels at lower level along x dir
    int sy; // Number of child voxels at lower level along y dir
    int sz; // Number of child voxels at lower level along z dir
    float dx; // Voxel size in x dir [mm]
	float dy; // Voxel size in y dir [mm]
	float dz; // Voxel size in z dir [mm]
};

// 3D measure of isotropy: defined as the cube volume over the cube area (volume normalized to 1)
float isotropy(float dx, float dy, float dz) {return pow(float(dx*dy*dz),float(2.0/3.0))/(2.0*(dx*dy+dx*dz+dy*dz));}
// 2D measure of isotropy: defined as the square area over the square perimeter (area normalized to 1)
float isotropy(float dx, float dy) {return sqrt(dx*dy)/(2.0*(dx+dy));}

Level getHigherLevel(Level level) {
	Level highLevel;
	highLevel.L = level.L+1;
	float maxIsotropy=0.0;
	for (int sx=1; sx<=2; sx++)
		for (int sy=1; sy<=2; sy++)
			for (int sz=1; sz<=2; sz++) // Loop over all 2^3=8 downscaling combinations
				if (sx*sy*sz>1 && level.nx>=sx && level.ny>=sy && level.nz>=sz) { // at least one dimension must change and the size of all dimensions at lower level must permit any downscaling
					float iso;
					if (level.nx==1) // Use 2D isotropy (y,z)
						iso = isotropy(level.dy*sy,level.dz*sz);
					else if (level.ny==1) // Use 2D isotropy (x,z)
						iso = isotropy(level.dx*sx,level.dz*sz);
					else if (level.nz==1) // Use 2D isotropy (x,y)
						iso = isotropy(level.dx*sx,level.dy*sy);
					else // Use 3D isotropy
						iso = isotropy(level.dx*sx,level.dy*sy,level.dz*sz);
					if (iso>maxIsotropy) {
						maxIsotropy=iso;
						highLevel.sx=sx; highLevel.sy=sy; highLevel.sz=sz;
					}
				}
	highLevel.dx=level.dx*highLevel.sx; highLevel.dy=level.dy*highLevel.sy; highLevel.dz=level.dz*highLevel.sz;
	highLevel.nx=ceil(level.nx/float(highLevel.sx)); highLevel.ny=ceil(level.ny/float(highLevel.sy)); highLevel.nz=ceil(level.nz/float(highLevel.sz));
	return highLevel;
}

vector<RESIDUALTYPE> getSumVector(vector<vector<RESIDUALTYPE> > vectors, float scale) {
	vector<RESIDUALTYPE> sumVector(vectors[0].size());
	for (unsigned int k=0; k<vectors[0].size(); k++) {
		float sum = 0.0;
		for (unsigned int n=0; n<vectors.size(); n++)
			sum += vectors[n][k];
		sumVector[k] = sum*scale;
	}
	return sumVector;
}

void getHighLevelResidualImage(image<vector<RESIDUALTYPE> >* Jhigh,image<vector<RESIDUALTYPE> >* J,Level highLevel, Level level) {
	float scale = 1.0/highLevel.sx/highLevel.sy/highLevel.sz;
	for (int Z=0; Z<highLevel.nz; Z++)
		for (int Y=0; Y<highLevel.ny; Y++)
			for (int X=0; X<highLevel.nx; X++) {
				vector<vector<RESIDUALTYPE> > childs;
				int x = X*highLevel.sx;
				int y = Y*highLevel.sy;
				int z = Z*highLevel.sz;
				bool twoXchilds = highLevel.sx>1 && x+1<level.nx;
				bool twoYchilds = highLevel.sy>1 && y+1<level.ny;
				bool twoZchilds = highLevel.sz>1 && z+1<level.nz;
				childs.push_back(imRef(J,x,y,z));
				if (twoXchilds) childs.push_back(imRef(J,x+1,y,z));
				if (twoYchilds) childs.push_back(imRef(J,x,y+1,z));
				if (twoZchilds) childs.push_back(imRef(J,x,y,z+1));
				if (twoXchilds && twoYchilds) childs.push_back(imRef(J,x+1,y+1,z));
				if (twoXchilds && twoZchilds) childs.push_back(imRef(J,x+1,y,z+1));
				if (twoYchilds && twoZchilds) childs.push_back(imRef(J,x,y+1,z+1));
				if (twoXchilds && twoYchilds && twoZchilds) childs.push_back(imRef(J,x+1,y+1,z+1));
				imRef(Jhigh,X,Y,Z) = getSumVector(childs, scale);
			}
}

int getB0fromHighLevel(image<int>* dB0high,Level highLevel,int i,Level level) {
	int z = i/(level.nx*level.ny);
	int y = (i-level.nx*level.ny*z)/level.nx;
	int x = i-level.nx*(y+level.ny*z);
	return imRef(dB0high,x/highLevel.sx,y/highLevel.sy,z/highLevel.sz);
}


void calculateFieldMap(image<int>* dB0,int nB0,Level level,int graphcutLevel,bool multiScale,int maxICMupdate,int nICMiter,image<vector<RESIDUALTYPE> >* J,vector<float> V,float mu) {
	// Trivial case at coarsest level with only one voxel
	int numNodes = level.nx*level.ny*level.nz;
	if (numNodes==1) {
		cout << "Level (1,1,1):" << endl << "Trivial case...";
		dB0->data[0]=(findGlobalMinimum(J->data[0],nB0)).second;
		cout << "DONE" << endl;
		return;
	}

	// Get fieldmap (dB0high) at higher level recursively
	image<int>* dB0high=NULL;
	Level highLevel;
	if (multiScale) {
		highLevel = getHigherLevel(level);
		image<vector<RESIDUALTYPE> >* Jhigh = new image<vector<RESIDUALTYPE> >(highLevel.nx,highLevel.ny,highLevel.nz,false); // Residual image
		getHighLevelResidualImage(Jhigh,J,highLevel,level);
		dB0high = new image<int>(highLevel.nx,highLevel.ny,highLevel.nz,false); // Off-resonance index image
		calculateFieldMap(dB0high,nB0,highLevel,graphcutLevel,multiScale,maxICMupdate,nICMiter,Jhigh,V,mu); // Recursion
		delete Jhigh;
		cout << "Level ("<<level.nx<<","<<level.ny<<","<<level.nz<<"): " << endl;
	}

	///// QPBO
	cout << "Preparing MRF...";

	image<int>* min1 = new image<int>(level.nx,level.ny,level.nz,false);
	image<int>* min2 = new image<int>(level.nx,level.ny,level.nz,false);
	findTwoSsmallestMinima(J, min1, min2, nB0);

	image<RESIDUALTYPE>* ddJ = new image<RESIDUALTYPE>(level.nx,level.ny,level.nz,false); // 2nd derivative image
	find2ndDerivativeAtMinimum(J,ddJ,min1,nB0,level.nx,level.ny,level.nz);
	image<float>* wx = new image<float>(level.nx,level.ny,level.nz,false);
	image<float>* wy = new image<float>(level.nx,level.ny,level.nz,false);
	image<float>* wz = new image<float>(level.nx,level.ny,level.nz,false);
	calcWeights(wx,wy,wz,ddJ,mu,level.dx,level.dy,level.dz,level.nx,level.ny,level.nz);
	delete ddJ;

	cout << "DONE" << endl;

	QPBO<MRFTYPE>* MRF = NULL;
	bool graphcut = level.L>=graphcutLevel;
	if (graphcut) {
		cout << "Solving MRF using QPBO...";
		int numEdges = (level.nx-1)*level.ny*level.nz+level.nx*(level.ny-1)*level.nz+level.nx*level.ny*(level.nz-1);
		MRF = new QPBO<MRFTYPE>(numNodes,numEdges);
		MRF->AddNode(numNodes); // add all nodes
		addTermsToMRF(MRF,J,min1,min2,wx,wy,wz,V,level.nx,level.ny,level.nz);
		MRF->Solve(); // Run QPBO
		cout << "DONE" << endl;
	}

	for (int i=0; i<numNodes; i++) {
		if (graphcut && MRF->GetLabel(i)==1) dB0->data[i]=min2->data[i];
		else if (graphcut && MRF->GetLabel(i)==0) dB0->data[i]=min1->data[i];
		else if (multiScale) dB0->data[i]=getB0fromHighLevel(dB0high,highLevel,i,level); // Get from higher level if not determined
		else dB0->data[i]=min1->data[i]; // If not determined, use smallest minimum
	}
	if (multiScale) delete dB0high;
	if (graphcut) delete MRF;

	delete min1;
	delete min2;

	if (nICMiter>0)	{
		cout << "Solving MRF using ICM...";
		ICM(dB0,nB0,maxICMupdate,nICMiter,J,V,wx,wy,wz,level.nx,level.ny,level.nz);
		cout << "DONE" << endl;
	}
	delete wx;
	delete wy;
	delete wz;
}

extern "C" {
__declspec(dllexport) void __cdecl fwqpbo(const IMGTYPE* Yreal,const IMGTYPE* Yimag,int N,int nx,int ny,int nz,float dx,float dy,float dz,float t1,float dt,float B0,float* CS,float* alpha,int M,int P,float R2step,int nR2,int* iR2cand,int nR2cand,bool FibSearch,float mu,int nB0,int nICMiter,int maxICMupdate,int graphcutLevel,bool multiScale,IMGTYPE* Xreal,IMGTYPE* Ximag,IMGTYPE* R2map,IMGTYPE* B0map)
{
	// ---------- PREPARE AND PRECALCULATE ---------- //
	cout << "Preparations and precalculations...";
	bool determineB0 = graphcutLevel<20 || nICMiter>0;
	bool determineR2 = nR2>1; // Estimate R2 and save into R2map
	if (nR2<0) nR2 = -nR2; // nR2<(-1) will use input R2map

	if (nR2==0) nR2=1; // Interpreted as no R2* mapping, i.e. R2*=0 and nR2=1
	if (FibSearch && determineR2) nR2 = get_nearest_higher_Fibonacci_number(nR2-1)+1; // Adjust nR2 to Fib. number+1 if needed
	for (int r=0; r<nR2cand; r++) if (iR2cand[r]<0 or iR2cand[r]>nR2-1) throw runtime_error("R2 candidate index out of range");

	// Precalculate lower triangular indices
    vector<vector<int> > lowTriInd(N,vector<int>(N));
    for (int m=0; m<N; m++) for (int n=0; n<=m; n++) lowTriInd[m][n] = (m+1)*m/2+n;
    int vecLen = (N+1)*N/2; // Length of matrices on "lower triangular vector form"

	image<VectorXcf>* S = new image<VectorXcf>(nx,ny,nz,false);
	createEchoImages(S,Yreal,Yimag,N,nx,ny,nz); 			    // Put raw data into complex vector image

	MatrixXcf A = modelMatrix(t1, dt, B0, N, CS, alpha, M, P);  // Model matrix
	vector<MatrixXcf> R = decayMatrices(nR2, R2step, N, t1, dt);// R2* decay matrices
    vector<MatrixXcf> B = modulationMatrices(nB0, N);           // B0 off-resonance modulation matrices

    vector<vector<MatrixXcf> > Qp(nB0, vector<MatrixXcf>(nR2)); // Pseudoinverses of Q=B*R*A
    vector<vector<MatrixXcf> > C(nB0, vector<MatrixXcf>(nR2));  // Null space projection matrices
    vector<vector<VectorXcf> > Cvec(nB0, vector<VectorXcf>(nR2));// Vector representation of C for efficient residual calculation

    for (int b=0; b<nB0; b++)
        for (int r=0; r<nR2; r++) {
            MatrixXcf Q = B[b]*R[r]*A;
            Qp[b][r] = pseudoinverse(Q);
            C[b][r]=MatrixXcf::Identity(N,N)-Q*Qp[b][r];

            Cvec[b][r]=VectorXcf(vecLen);
            for (int m=0; m<N; m++)
                for (int n=0; n<=m; n++) {
                    if (m==n) Cvec[b][r](lowTriInd[m][n]) = C[b][r](m,n);
                    else Cvec[b][r](lowTriInd[m][n]) = complex<float>(2,0)*C[b][r](m,n); // Factor 2 for lower triangular entries for efficient multiplication utilizing Hermitean symmetry
                }
        }

	vector<float> V(nB0); //Precalculate discontinuity costs
	// NOTE: No multiplication of square(steplength) since no division of square(steplength) in ddJ (J'')
	for (int b=0; b<nB0; b++) V[b] = min(square(b),square(b-nB0));

	cout << "DONE" << endl;

	image<int>* dB0 = new image<int>(nx,ny,nz,false); // Off-resonance index image
	float B0step = 1.0/nB0/dt/gyro/B0; // For converting B0 index to off-resonance in ppm
	if (determineB0) {
        // ---------- RESIDUAL CALCULATIONS ---------- //
        cout << "Residual calculations..." << endl;

        image<vector<RESIDUALTYPE> >* J = new image<vector<RESIDUALTYPE> >(nx,ny,nz,false); // Residual image
        getResidualImages(J,S,Cvec,lowTriInd,N,nB0,iR2cand,nR2cand,nx*ny*nz);

        cout << "DONE" << endl;

        // ---------- CALCULATE B0 FIELD MAP ---------- //
        cout << "Calculating B0 field map:" << endl;

        Level level;
        level.L=0;
        level.nx=nx; level.ny=ny; level.nz=nz;
        level.sx=1; level.sy=1; level.sz=1;
        level.dx=dx; level.dy=dy; level.dz=dz;
        calculateFieldMap(dB0,nB0,level,graphcutLevel,multiScale,maxICMupdate,nICMiter,J,V,mu);
        delete J;

        cout << "DONE" << endl;
    }
    else
    	for (int i=0; i<nx*ny*nz; i++)
            dB0->data[i] = int(B0map[i]/B0step);

	image<int>* R2=NULL;
	if (determineR2) {
	// ---------- ESTIMATE R2* ---------- //
		cout << "Estimating R2*-map...";

		R2 = new image<int>(nx,ny,nz,false); // R2* index image
		for (int i=0; i<nx*ny*nz; i++) {
			if (nR2==1) R2->data[i] = 0;
			else {
				if (FibSearch && nR2>2) R2->data[i] = FibonacciSearch(S->data[i], Cvec[dB0->data[i]], nR2, N, lowTriInd);
				else R2->data[i] = ExhaustiveSearch(S->data[i], Cvec[dB0->data[i]], nR2, N, lowTriInd);
			}
		}
		cout << "DONE" << endl;
	}

	// ---------- SOLVE LEAST SQUARES ---------- //
	cout << "Finding least-squares solution...";

	for (int i=0; i<nx*ny*nz; i++) {
        int iB0 = dB0->data[i];
        B0map[i]=IMGTYPE(iB0*B0step);

		int iR2;
		if (nR2>1) {
            if (determineR2) {
                iR2 = R2->data[i];
                R2map[i]=IMGTYPE(iR2*R2step);
            }
            else iR2 = int(R2map[i]/R2step);
		}
		else iR2 = 0;
        VectorXcf X = Qp[iB0][iR2]*S->data[i];
        for (int m=0; m<M; m++) {
            Xreal[i+m*nx*ny*nz]=IMGTYPE(real(X[m]));
            Ximag[i+m*nx*ny*nz]=IMGTYPE(imag(X[m]));
        }
	}
	cout << "DONE" << endl;

	delete dB0;
	if (determineR2) delete R2;
	delete S;
	return;
}
}
