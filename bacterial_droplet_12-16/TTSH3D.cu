#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cufft.h>
#include <cufftXt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <math.h>
#include <random> // The header for the generators.
#include <iomanip>

//=======================================================================
// Definitions ----------------------------------------------------------
// Define the precision of real numbers, could be float/double.
#define real double
#define Pi 3.1415926535897932384626433832795
#define Zero 0
typedef double2 Complex;

using namespace std;

// Initialize random number seed. To generate a random number, use:
// uniform_real_distribution<real> randUR; a=randUR(rng);
// or:
// uniform_int_distribution<int> randUI; a=randUI(rng);
using std::default_random_engine;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
int seed = time(0);
default_random_engine rng(seed);
int gpu_node = 1;
// Declare variables ----------------------------------------------------
struct Fields
{ // Flow fileds
  real *vx;
  real *vy;
  real *vz;
  real *vx_star;
  real *vy_star;
  real *vz_star;
  real *dtvx_star;
  real *dtvy_star;
  real *dtvz_star;
  real *p_v;
  
  // Polar fields
  real *P2;
  real *px;
  real *py;
  real *pz;
  real *px_star;
  real *py_star;
  real *pz_star;
  real *dtpx_star;
  real *dtpy_star;
  real *dtpz_star;      
  real *mu_x;
  real *mu_y;
  real *mu_z;
  real *dxPx;
  real *dyPx;
  real *dzPx;
  real *dxPy;
  real *dyPy;
  real *dzPy;
  real *dxPz;
  real *dyPz;
  real *dzPz;
  real *Sigxx;
  real *Sigxy;
  real *Sigxz;
  real *Sigyx;
  real *Sigyy;
  real *Sigyz;
  real *Sigzx;
  real *Sigzy;
  real *Sigzz;
  real *DPx;
  real *DPy;
  real *DPz;
  real *OPx;
  real *OPy;
  real *OPz;
  real *p_p;
  real *Axx;
  real *Axy;
  real *Axz;
  real *Ayy;
  real *Ayz;
  real *Azz;

  // Phase fields
  real *phi;
  real *dtphi;
  real *mu_0;
  real *dPhi2;

  cufftDoubleComplex *Fstream;
};

struct Parameters
{
  int TimeScheme;
  int RKStage;
  int ExplImpi;
  int AdapTime;
  int ExpoData;
  int InitCond;
  int isSimplified;
  int checkUnStable;
  int Nx;
  int Ny;
  int Nz;
  int Nb;
  real h;
  real dt0;
  real T0;
  real Ts;
  real dte;
  int adap_dt;
  real dtVarC;
  real dtMax;
  // Parameters for velocity field
  real mu;
  real cA;
  // Parameters for polar field
  real eta;
  real La;
  real alpha;
  real beta;
  real lambda0;
  real kappa;
  real A;
  // Parameters for phase field
  real aPhi;
  real kPhi;
  real gamma;
  // Initial conditions
  real init0;

  // real dQxy0;
  // Other parameters
  int GSize;
  int BSize;
  int ReadSuccess;
};

struct ParaRKC2
{
  real *mu;
  real *mu1;
  real *nu;
  real *gamma1;
} PRKC2, pRKC2;

struct Poisson
{
  real *k2;
} PS, ps;

cufftHandle plan;

int iStop;
real T;
real Dt;
real progress;
real *t;
real *dt;

// Use F/f to store fields in host/device.
Fields F, f, f1, f2, f3;
// Similarily, we use P/p to store parameters in host/device.
Parameters P;
__constant__ Parameters p;
// Declare other global run-time variables.
real *dtVarMX;
real *dtVarM;
real DtVarM;
clock_t tStart;

// Declare functions ----------------------------------------------------
// Host functions
void GetInput();
void InitConf();
void evolve();
void run();
void EEuler();
void EPredCorr();
void RKC2();
void ERK4();
void getRHS(Fields ff);
void getP(Fields ff, int fieldType);
void testHost();
void initRandSeed();
void initPoissonSolver();
void ShowProgress();
void getPRKC2();
void ExpoConf(string str_t);
void MemAlloc();
void MemFree();
void MemAllocFields(Fields ff);
void MemFreeFields(Fields ff);

// Device functions
__global__ void initRandSeedDevi(unsigned int seed, curandState_t *states);
__global__ void getStream(Fields ff, Poisson ps1, int getType, int fieldType);
__global__ void getPCore(Fields ff);
__global__ void getTempVari_1(Fields ff);
__global__ void getTempVari_2(Fields ff);
__global__ void getDtF(Fields ff);
__global__ void copyF(Fields ff1, Fields ff2);
// __global__ void update(Fields ff1, Fields ff2, Fields ff3, real ddt, int getType);
__global__ void getVStar(Fields ff1, Fields ff2, Fields ff3, real ddt);
__global__ void getVNew(Fields ff1, Fields ff2, Fields ff3, real ddt);
__global__ void getDivVStar(Fields ff1, Fields ff2, Fields ff3, real ddt);
__global__ void BounPeriF(Fields ff);
__global__ void BounPeriFT(Fields ff);
__global__ void BounPeriU(real *u);
__global__ void getMaxX(Fields, real *dtVarMX);
__global__ void getDt(real *dtVarMX, real *dtVarM, real *dt);
__global__ void testDevi(Fields ff);
__device__ real d1xO2I(real *u, int i, int j);
__device__ real d1yO2I(real *u, int i, int j);
__device__ real d2xO2I(real *u, int i, int j);
__device__ real d2yO2I(real *u, int i, int j);
__device__ real d1x1yO2I(real *u, int i, int j);
__device__ real LaplO2I(real *u, int i, int j);
__device__ real BiLaO2I(real *u, int i, int j);
__device__ real d1xO4I(real *u, int i, int j);
__device__ real d1yO4I(real *u, int i, int j);
__device__ real d2xO4I(real *u, int i, int j);
__device__ real d2yO4I(real *u, int i, int j);
__device__ real d1x1yO4I(real *u, int i, int j);
__device__ real LaplO4I(real *u, int i, int j);
__device__ real BiLaO4I(real *u, int i, int j);
__device__ real d1xO4(real *u, int i, int j);
__device__ real d1yO4(real *u, int i, int j);
__device__ real d2xO4(real *u, int i, int j);
__device__ real d2yO4(real *u, int i, int j);
__device__ real d1x1yO4(real *u, int i, int j);
__device__ real LaplO4(real *u, int i, int j);
__device__ real BiLaO4(real *u, int i, int j);
__device__ real d2x2yO4(real *u, int i, int j);
__device__ real d4xO4(real *u, int i, int j);
__device__ real d4yO4(real *u, int i, int j);
__device__ real d1xCO2I3D(real *u, int i, int j, int k);
__device__ real d1yCO2I3D(real *u, int i, int j, int k);
__device__ real d1zCO2I3D(real *u, int i, int j, int k);
__device__ real laplaceCO2I3D(real *u, int i, int j, int k);

//=======================================================================
int main()
{
  // Get starting time of simulation.
  tStart = clock();
  cudaDeviceReset();
  // Get parameters from file.
  GetInput();
  if (P.ReadSuccess == -8848)
  {
    // Add a part that check if old data files exist and delete them.
    initRandSeed();
    MemAlloc();
    initPoissonSolver();
    InitConf();
    // dim3 grid(64,64);
    // dim3 block(64,1);
    evolve();
    cufftDestroy(plan);
    MemFree();
  }

  clock_t tEnd = clock();
  cout << "Simulation finished. ";
  cout << "CPU time = " << double(tEnd - tStart) / CLOCKS_PER_SEC << " sec" << endl;
}

//=======================================================================
void evolve()
{
  std::string str_t;
  // This part is for testing --------------------------------------------
  // BounPeriF<<<P.Ny,P.Nx>>>(f);
  // testHost();
  // for (int i=0;i<10000;i++) {
  // getV(f);
  // }
  // iStop=1;
  // This part is for testing --------------------------------------------
  ShowProgress();
  if (P.T0 == 0)
  {
    ExpoConf("0");
  }
  if (P.TimeScheme == 3)
  {
    getPRKC2();
  }

  while (iStop == 0)
  {

    // Evolution
    run();
    // Export.
    if (floor(T / P.dte) > floor((T - Dt) / P.dte))
    {

      int te = floor(T / P.dte);
      str_t = to_string(te);
      ExpoConf(str_t);
      ShowProgress();
    }

    // Check.
    if (T > P.Ts)
    {
      iStop = 1;
    }
  }
}


//=======================================================================
void run()
{
  if (P.TimeScheme == 1)
  {
    EEuler();
  }
  T = T + Dt;
}


// Explicit Euler Scheme ================================================
void EEuler()
{
  getRHS(f);

  getVStar<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(f, f, f, Dt);

  getP(f,0);
  getP(f,1);
 
  getVNew<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(f, f, f, Dt);


}


//=======================================================================
void getRHS(Fields ff)
{
  // BounPeriF<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff);;
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.vx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.vy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.vz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.vx_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.vy_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.vz_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.phi);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_0);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_x);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_y);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_z);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtphi);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtvx_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtvy_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtvz_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtpx_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtpy_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dtpz_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigxx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigxy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigxz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigyx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigyy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigyz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigzx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigzy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigzz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.DPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.DPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.DPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.OPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.OPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.OPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.p_v);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.p_p);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.P2);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.px);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.py);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.pz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.px_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.py_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.pz_star);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dxPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dyPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dzPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dxPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dyPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dzPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dxPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dyPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dzPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Axx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Axy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Axz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Ayy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Ayz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Azz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dPhi2);
  
  getTempVari_1<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff);

  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dxPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dyPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dzPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dxPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dyPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dzPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dxPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dyPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dzPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.dPhi2);
  
  getTempVari_2<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff);
  
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_0);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_x);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_y);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.mu_z);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.P2);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.DPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.DPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.DPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.OPx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.OPy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.OPz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigxx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigxy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigxz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigyx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigyy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigyz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigzx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigzy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Sigzz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Axx);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Axy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Axz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Ayy);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Ayz);
  BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.Azz);
  
  getDtF<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff);
}

//=======================================================================
__global__ void getVStar(Fields ff1, Fields ff2, Fields ff3, real ddt)
{
  // ff1=ff2+dt*ff3
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  // int idx = (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;

  if (i < p.Ny && j < p.Nx && k < p.Nz)
  {

      ff1.vx_star[idx] = ff2.vx[idx] + ddt * ff3.dtvx_star[idx];
      ff1.vy_star[idx] = ff2.vy[idx] + ddt * ff3.dtvy_star[idx];
      ff1.vz_star[idx] = ff2.vz[idx] + ddt * ff3.dtvz_star[idx];

      ff1.px_star[idx] = ff2.px[idx] + ddt * ff3.dtpx_star[idx];
      ff1.py_star[idx] = ff2.py[idx] + ddt * ff3.dtpy_star[idx];
      ff1.pz_star[idx] = ff2.pz[idx] + ddt * ff3.dtpz_star[idx];
  }
}

//=======================================================================
__global__ void getVNew(Fields ff1, Fields ff2, Fields ff3, real ddt)
{
  // ff1=ff2+dt*ff3
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;

  if (i < p.Ny && j < p.Nx && k < p.Nz)
  {

      ff1.vx[idx] = ff2.vx_star[idx] - ddt * d1xCO2I3D(ff3.p_v, i, j, k);
      ff1.vy[idx] = ff2.vy_star[idx] - ddt * d1yCO2I3D(ff3.p_v, i, j, k);
      ff1.vz[idx] = ff2.vz_star[idx] - ddt * d1zCO2I3D(ff3.p_v, i, j, k);

      ff1.px[idx] = ff2.px_star[idx] - ddt * d1xCO2I3D(ff3.p_p, i, j, k);
      ff1.py[idx] = ff2.py_star[idx] - ddt * d1yCO2I3D(ff3.p_p, i, j, k);
      ff1.pz[idx] = ff2.pz_star[idx] - ddt * d1zCO2I3D(ff3.p_p, i, j, k);

      ff1.phi[idx] = ff2.phi[idx] + ddt * ff3.dtphi[idx];

  }
}

//=======================================================================
__global__ void copyF(Fields ff1, Fields ff2)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  // int idx = (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;

  if (i < p.Ny && j < p.Nx && k < p.Nz)
  {

    ff1.p_v[idx] = ff2.p_v[idx];
    ff1.p_p[idx] = ff2.p_p[idx];

  }
}

//=======================================================================
__global__ void getTempVari_1(Fields ff)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  // int idx = (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  if (i < p.Ny && j < p.Nx && k < p.Nz)
  {
    ff.dxPx[idx] = d1xCO2I3D(ff.px, i, j, k);
    ff.dyPx[idx] = d1yCO2I3D(ff.px, i, j, k);
    ff.dzPx[idx] = d1zCO2I3D(ff.px, i, j, k);
    ff.dxPy[idx] = d1xCO2I3D(ff.py, i, j, k);
    ff.dyPy[idx] = d1yCO2I3D(ff.py, i, j, k);
    ff.dzPy[idx] = d1zCO2I3D(ff.py, i, j, k);
    ff.dxPz[idx] = d1xCO2I3D(ff.pz, i, j, k);
    ff.dyPz[idx] = d1yCO2I3D(ff.pz, i, j, k);
    ff.dzPz[idx] = d1zCO2I3D(ff.pz, i, j, k);

    ff.dPhi2[idx] = d1xCO2I3D(ff.phi, i, j, k) * d1xCO2I3D(ff.phi, i, j, k) + d1yCO2I3D(ff.phi, i, j, k) * d1yCO2I3D(ff.phi, i, j, k) + d1zCO2I3D(ff.phi, i, j, k) * d1zCO2I3D(ff.phi, i, j, k);
  }
}

//=======================================================================
__global__ void getTempVari_2(Fields ff)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;

  if (i < p.Ny && j < p.Nx && k < p.Nz)
  {
    
    ff.mu_0[idx] = p.aPhi*2*(ff.phi[idx] - 3*ff.phi[idx]*ff.phi[idx] + 2*ff.phi[idx]*ff.phi[idx]*ff.phi[idx])-p.kPhi*laplaceCO2I3D(ff.phi, i, j, k);
    
    ff.P2[idx] = ff.px[idx]*ff.px[idx] + ff.py[idx]*ff.py[idx] + ff.pz[idx]*ff.pz[idx];

    ff.mu_x[idx] = ff.phi[idx]*ff.px[idx] + p.La*p.La * laplaceCO2I3D(ff.px, i, j, k);
    ff.mu_y[idx] = ff.phi[idx]*ff.py[idx] + p.La*p.La * laplaceCO2I3D(ff.py, i, j, k);
    ff.mu_z[idx] = ff.phi[idx]*ff.pz[idx] + p.La*p.La * laplaceCO2I3D(ff.pz, i, j, k);

    ff.DPx[idx] = 1/2*(ff.px[idx]*2*d1xCO2I3D(ff.vx, i, j ,k) + ff.py[idx]*(d1xCO2I3D(ff.vy, i, j, k) + d1yCO2I3D(ff.vx, i, j, k)) + ff.pz[idx]*(d1xCO2I3D(ff.vz, i, j, k) + d1zCO2I3D(ff.vx, i, j, k)));
    ff.DPy[idx] = 1/2*(ff.px[idx]*(d1yCO2I3D(ff.vx, i, j, k) + d1xCO2I3D(ff.vy, i, j, k)) + ff.py[idx]*2*d1yCO2I3D(ff.vy, i, j, k) + ff.pz[idx]*(d1yCO2I3D(ff.vz, i, j, k) + d1zCO2I3D(ff.vy, i, j, k)));
    ff.DPz[idx] = 1/2*(ff.px[idx]*(d1zCO2I3D(ff.vx, i, j, k) + d1xCO2I3D(ff.vz, i, j, k)) + ff.py[idx]*(d1zCO2I3D(ff.vy, i, j, k) + d1yCO2I3D(ff.vz, i, j, k)) + ff.pz[idx]*2*d1zCO2I3D(ff.vz, i, j, k));

    ff.OPx[idx] = 1/2*(ff.py[idx] * (d1xCO2I3D(ff.vy, i, j, k) - d1yCO2I3D(ff.vx, i, j, k)) + ff.pz[idx] * (d1xCO2I3D(ff.vz, i, j, k) - d1zCO2I3D(ff.vx, i, j, k)) );
    ff.OPy[idx] = 1/2*(ff.pz[idx] * (d1yCO2I3D(ff.vz, i, j, k) - d1zCO2I3D(ff.vy, i, j, k)) + ff.px[idx] * (d1yCO2I3D(ff.vx, i, j, k) - d1xCO2I3D(ff.vy, i, j, k)) );
    ff.OPz[idx] = 1/2*(ff.px[idx] * (d1zCO2I3D(ff.vx, i, j, k) - d1xCO2I3D(ff.vz, i, j, k)) + ff.py[idx] * (d1zCO2I3D(ff.vy, i, j, k) - d1yCO2I3D(ff.vz, i, j, k)) );

    ff.Sigxx[idx] = ff.phi[idx] * (ff.dxPx[idx] + 1/28*laplaceCO2I3D(ff.dxPx, i, j, k));
    ff.Sigxy[idx] = ff.phi[idx] * (ff.dxPy[idx] + 1/28*laplaceCO2I3D(ff.dxPy, i, j, k));
    ff.Sigxz[idx] = ff.phi[idx] * (ff.dxPz[idx] + 1/28*laplaceCO2I3D(ff.dxPz, i, j, k));
    ff.Sigyx[idx] = ff.phi[idx] * (ff.dyPx[idx] + 1/28*laplaceCO2I3D(ff.dyPx, i, j, k));
    ff.Sigyy[idx] = ff.phi[idx] * (ff.dyPy[idx] + 1/28*laplaceCO2I3D(ff.dyPy, i, j, k));
    ff.Sigyz[idx] = ff.phi[idx] * (ff.dyPz[idx] + 1/28*laplaceCO2I3D(ff.dyPz, i, j, k));
    ff.Sigzx[idx] = ff.phi[idx] * (ff.dzPx[idx] + 1/28*laplaceCO2I3D(ff.dzPx, i, j, k));
    ff.Sigzy[idx] = ff.phi[idx] * (ff.dzPy[idx] + 1/28*laplaceCO2I3D(ff.dzPy, i, j, k));
    ff.Sigzz[idx] = ff.phi[idx] * (ff.dzPz[idx] + 1/28*laplaceCO2I3D(ff.dzPz, i, j, k));

    ff.Axx[idx] = d1xCO2I3D(ff.phi, i, j, k) * d1xCO2I3D(ff.phi, i, j, k) - 1/3*ff.dPhi2[idx];
    ff.Axy[idx] = d1xCO2I3D(ff.phi, i, j, k) * d1yCO2I3D(ff.phi, i, j, k);
    ff.Axz[idx] = d1xCO2I3D(ff.phi, i, j, k) * d1zCO2I3D(ff.phi, i, j, k);
    ff.Ayy[idx] = d1yCO2I3D(ff.phi, i, j, k) * d1yCO2I3D(ff.phi, i, j, k) - 1/3*ff.dPhi2[idx];
    ff.Ayz[idx] = d1yCO2I3D(ff.phi, i, j, k) * d1zCO2I3D(ff.phi, i, j, k);
    ff.Azz[idx] = d1zCO2I3D(ff.phi, i, j, k) * d1zCO2I3D(ff.phi, i, j, k) - 1/3*ff.dPhi2[idx];

  }
}

//=======================================================================
__global__ void getDtF(Fields ff)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  // int idx = (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  if (i < p.Ny && j < p.Nx && k < p.Nz)
  
  {
    ff.dtvx_star[idx] = -ff.vx[idx]*d1xCO2I3D(ff.vx, i, j, k) - ff.vy[idx]*d1yCO2I3D(ff.vx, i, j, k) - ff.vz[idx]*d1zCO2I3D(ff.vx, i, j, k) + p.mu*laplaceCO2I3D(ff.vx, i, j, k) + ff.mu_0[idx] * d1xCO2I3D(ff.phi, i, j, k) + p.cA*(d1xCO2I3D(ff.Sigxx, i, j, k) + d1yCO2I3D(ff.Sigyx, i, j, k) + d1zCO2I3D(ff.Sigzx, i, j, k));

    ff.dtvy_star[idx] = -ff.vx[idx]*d1xCO2I3D(ff.vy, i, j, k) - ff.vy[idx]*d1yCO2I3D(ff.vy, i, j, k) - ff.vz[idx]*d1zCO2I3D(ff.vy, i, j, k) + p.mu*laplaceCO2I3D(ff.vy, i, j, k) + ff.mu_0[idx] * d1yCO2I3D(ff.phi, i, j, k) + p.cA*(d1xCO2I3D(ff.Sigxy, i, j, k) + d1yCO2I3D(ff.Sigyy, i, j, k) + d1zCO2I3D(ff.Sigzy, i, j, k));
  
    ff.dtvz_star[idx] = -ff.vx[idx]*d1xCO2I3D(ff.vz, i, j, k) - ff.vy[idx]*d1yCO2I3D(ff.vz, i, j, k) - ff.vz[idx]*d1zCO2I3D(ff.vz, i, j, k) + p.mu*laplaceCO2I3D(ff.vz, i, j, k) + ff.mu_0[idx] * d1zCO2I3D(ff.phi, i, j, k) + p.cA*(d1xCO2I3D(ff.Sigxz, i, j, k) + d1yCO2I3D(ff.Sigyz, i, j, k) + d1zCO2I3D(ff.Sigzz, i, j, k));
    
    ff.dtpx_star[idx] = -ff.vx[idx]*d1xCO2I3D(ff.px, i, j, k) - ff.vy[idx]*d1yCO2I3D(ff.px, i, j, k) - ff.vz[idx]*d1zCO2I3D(ff.px, i, j, k) - p.alpha*ff.phi[idx]*ff.px[idx] - p.beta * ff.P2[idx] * ff.px[idx] + p.eta*laplaceCO2I3D(ff.mu_x, i, j, k) + ff.OPx[idx] + p.kappa * ff.phi[idx]*ff.DPx[idx] - p.lambda0 * (ff.px[idx]*d1xCO2I3D(ff.px, i, j, k) + ff.py[idx]*d1yCO2I3D(ff.px, i, j, k) + ff.pz[idx]*d1zCO2I3D(ff.px, i, j, k)) - p.A * (ff.dPhi2[idx] * (ff.px[idx]*ff.px[idx]*ff.px[idx]+ff.px[idx]*ff.py[idx]*ff.py[idx]+ff.px[idx]*ff.pz[idx]*ff.pz[idx])+ sqrt(ff.P2[idx]) * (ff.Axx[idx]*ff.px[idx] + ff.Axy[idx]*ff.py[idx] + ff.Axz[idx]*ff.pz[idx]));

    ff.dtpy_star[idx] = -ff.vx[idx]*d1xCO2I3D(ff.py, i, j, k) - ff.vy[idx]*d1yCO2I3D(ff.py, i, j, k) - ff.vz[idx]*d1zCO2I3D(ff.py, i, j, k) - p.alpha*ff.phi[idx]*ff.py[idx] - p.beta * ff.P2[idx] * ff.py[idx] + p.eta*laplaceCO2I3D(ff.mu_y, i, j, k) + ff.OPy[idx] + p.kappa * ff.phi[idx]*ff.DPy[idx] - p.lambda0 * (ff.px[idx]*d1xCO2I3D(ff.py, i, j, k) + ff.py[idx]*d1yCO2I3D(ff.py, i, j, k) + ff.pz[idx]*d1zCO2I3D(ff.py, i, j, k)) - p.A * (ff.dPhi2[idx] * (ff.px[idx]*ff.py[idx]*ff.px[idx]+ff.py[idx]*ff.py[idx]*ff.py[idx]+ff.py[idx]*ff.pz[idx]*ff.pz[idx])+ sqrt(ff.P2[idx]) * (ff.Axy[idx]*ff.px[idx] + ff.Ayy[idx]*ff.py[idx] + ff.Ayz[idx]*ff.pz[idx]));

    ff.dtpz_star[idx] = -ff.vx[idx]*d1xCO2I3D(ff.pz, i, j, k) - ff.vy[idx]*d1yCO2I3D(ff.pz, i, j, k) - ff.vz[idx]*d1zCO2I3D(ff.pz, i, j, k) - p.alpha*ff.phi[idx]*ff.pz[idx] - p.beta * ff.P2[idx] * ff.pz[idx] + p.eta*laplaceCO2I3D(ff.mu_z, i, j, k) + ff.OPz[idx] + p.kappa * ff.phi[idx]*ff.DPz[idx] - p.lambda0 * (ff.px[idx]*d1xCO2I3D(ff.pz, i, j, k) + ff.py[idx]*d1yCO2I3D(ff.pz, i, j, k) + ff.pz[idx]*d1zCO2I3D(ff.pz, i, j, k)) - p.A * (ff.dPhi2[idx] * (ff.px[idx]*ff.pz[idx]*ff.px[idx]+ff.py[idx]*ff.pz[idx]*ff.py[idx]+ff.pz[idx]*ff.pz[idx]*ff.pz[idx])+ sqrt(ff.P2[idx]) * (ff.Axz[idx]*ff.px[idx] + ff.Ayz[idx]*ff.py[idx] + ff.Azz[idx]*ff.pz[idx])); 

    ff.dtphi[idx] = - ff.vx[idx]*d1xCO2I3D(ff.phi, i,j ,k) - ff.vy[idx]*d1yCO2I3D(ff.phi, i,j ,k) - ff.vz[idx]*d1zCO2I3D(ff.phi, i,j ,k) + p.gamma * (laplaceCO2I3D(ff.mu_0, i, j, k));
  }
}

//=======================================================================
void getP(Fields ff, int fieldType)
{
  getStream<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff, ps, 0, fieldType);
  cufftExecZ2Z(plan, ff.Fstream, ff.Fstream, CUFFT_FORWARD);
  getStream<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff, ps, 1, fieldType);
  cufftExecZ2Z(plan, ff.Fstream, ff.Fstream, CUFFT_INVERSE);
  getStream<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff, ps, 2, fieldType);
  if (fieldType == 0)
  {
    BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.p_v);
  }
  else if (fieldType == 1)
  {
    BounPeriU<<<dim3(P.Ny,P.Nz), dim3(P.Nx, 1)>>>(ff.p_p);
  }

}

//========================================================================
__global__ void getStream(Fields ff, Poisson ps1, int getType, int fieldType)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx1 = (blockDim.x + 2 * p.Nb) * (gridDim.x + 2 * p.Nb) * (k + p.Nb) + (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int idx = (p.Nx * p.Ny) * k + i * p.Nx + j;

  if (getType == 0)
  {
    if (fieldType == 0) // 0=Flow
      ff.Fstream[idx].x = (d1xCO2I3D(ff.vx_star, i, j, k) + d1yCO2I3D(ff.vy_star, i, j, k) + d1zCO2I3D(ff.vz_star, i, j, k) )/p.dt0;
    else if (fieldType == 1) // 1=Polar
      ff.Fstream[idx].x = (d1xCO2I3D(ff.px_star, i, j, k) + d1yCO2I3D(ff.py_star, i, j, k) + d1zCO2I3D(ff.pz_star, i, j, k) )/p.dt0;

    ff.Fstream[idx].y = 0;
  }
  else if (getType == 1)
  {
    if (ps1.k2[idx] == 0)
    {
      ff.Fstream[idx].x = 0;
      ff.Fstream[idx].y = 0; // Setting ifft of w to 0 when wavenumber is 0
    }
    else
    {
      ff.Fstream[idx].x = -ff.Fstream[idx].x / (ps1.k2[idx]);
      ff.Fstream[idx].y = -ff.Fstream[idx].y / (ps1.k2[idx]);
    }
  }
  else if (getType == 2)
  {
    if (fieldType == 0) // 0=Flow
      ff.p_v[idx1] = ff.Fstream[idx].x / (p.Nx * p.Ny * p.Nz);
    else if (fieldType == 1) // 1=Polar
      ff.p_p[idx1] = ff.Fstream[idx].x / (p.Nx * p.Ny * p.Nz);
  }
}



//=======================================================================
void GetInput()
{
  ifstream InputFile("data_" + to_string(gpu_node) + "/input.dat");

  P.ReadSuccess = 0;
  // Simulation parameters
  InputFile >> P.TimeScheme;
  InputFile >> P.RKStage;
  InputFile >> P.ExplImpi;
  InputFile >> P.AdapTime;
  InputFile >> P.ExpoData;
  InputFile >> P.InitCond;
  InputFile >> P.isSimplified;
  InputFile >> P.checkUnStable;
  InputFile >> P.Nx;
  InputFile >> P.Ny;
  InputFile >> P.Nz;
  InputFile >> P.Nb;
  InputFile >> P.h;
  InputFile >> P.dt0;
  InputFile >> P.T0;
  InputFile >> P.Ts;
  InputFile >> P.dte;
  InputFile >> P.dtVarC;
  InputFile >> P.dtMax;

  // Model parameters
  InputFile >> P.mu;
  InputFile >> P.cA;
  InputFile >> P.alpha;
  InputFile >> P.beta;
  InputFile >> P.eta;
  InputFile >> P.La;
  InputFile >> P.kappa;
  InputFile >> P.lambda0;
  InputFile >> P.aPhi;
  InputFile >> P.kPhi;
  InputFile >> P.gamma;
  InputFile >> P.A;
  InputFile >> P.init0;

  // InputFile >> P.gpu_node;

  InputFile >> P.ReadSuccess;

  // Grid size.
  P.GSize = (P.Nx + 2 * P.Nb) * (P.Ny + 2 * P.Nb) * (P.Nz + 2 * P.Nb);
  // Byte size of the fields.
  P.BSize = P.GSize * sizeof(real);

  InputFile.close();
  cout << "hhhhhhhhhhh:"<<P.ReadSuccess << endl;
  if (P.ReadSuccess != -8848)
  {
    cout << "Error while reading the input file!" << endl;
  }
}

//=======================================================================
void InitConf()
{
  T = P.T0;
  iStop = 0;
  Dt = P.dt0;
  // Copy parameters from host memory to device memory
  cudaMemcpyToSymbol(p, &P, sizeof(Parameters));
  cudaMemcpy(t, &T, sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(dt, &Dt, sizeof(real), cudaMemcpyHostToDevice);

  uniform_real_distribution<real> randUR;
  // uniform_real_distribution<real> randUR; a=randUR(rng);
  int idx;
  int t0 = (P.T0 + 0.1 * P.dte) / P.dte;
  std::string InitFileName = "data/conf_" +
                             to_string(t0) + ".dat";
  // cout << InitFileName << endl;

  ifstream InputFile(InitFileName);
  std::string gammavTypeName = "gammav-Init"+to_string(P.InitCond)+"-Nx"+to_string(P.Nx)+"-Ny"+to_string(P.Ny);
  std::string gammavFileName = "../gammav-directory/" + gammavTypeName + "/" + gammavTypeName + ".txt";
  ifstream gammavInputFile(gammavFileName);
  if(P.InitCond > 100){
    cout << gammavFileName << endl;
    if(!gammavInputFile.is_open()){
      cout << " gammavFileName is not exist" << endl;
      return;
    }
  }
  

  for (int i = 0; i < P.Nx; i++)
  {
    for (int j = 0; j < P.Ny; j++)
    {
      for (int k = 0; k < P.Nz; k++)
      {
        // idx = (P.Nx + 2 * P.Nb) * (i + P.Nb) + j + P.Nb;
        idx = (P.Nx + 2 * P.Nb) * (P.Ny + 2 * P.Nb) * (k + P.Nb) + (P.Nx + 2 * P.Nb) * (j + P.Nb) + i + P.Nb;
        if (P.InitCond == 0)
        {
          InputFile >> F.vx[idx] >> F.vy[idx];
        }
        else if (P.InitCond == 1)
        {
          // F.vortex[idx] = cos(2 * Pi * i / P.Nx) * cos(4 * Pi * j / P.Ny) * cos(6 * Pi * k / P.Nz);
          F.vx[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.vy[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.vz[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.vx_star[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.vy_star[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.vz_star[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.px[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.py[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.pz[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.px_star[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.py_star[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.pz_star[idx] = 0 + P.init0 * (randUR(rng) - 0.5);

          real dx = (i - P.Nx / 2.);
          real dy = (j - P.Ny / 2.);
          real dz = (k - P.Nz / 2.);

          real r = sqrt(dx * dx + dy * dy + dz * dz);

          F.phi[idx] = (1 - tanh((r-P.Nx/4.)*2))/2;
                   

          F.p_v[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
          F.p_p[idx] = 0 + P.init0 * (randUR(rng) - 0.5);
        }
      }
    }
  }
  
  
  // Still need to get vortex
  // cudaMemcpy(f.vortex, F.vortex, P.BSize, cudaMemcpyHostToDevice);
  // cudaMemcpy(f.stream, F.stream, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.vx, F.vx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.vy, F.vy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.vz, F.vz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.vx_star, F.vx_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.vy_star, F.vy_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.vz_star, F.vz_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtvx_star, F.dtvx_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtvy_star, F.dtvy_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtvz_star, F.dtvz_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.p_v, F.p_v, P.BSize, cudaMemcpyHostToDevice);

  cudaMemcpy(f.phi, F.phi, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtphi, F.dtphi, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.mu_0, F.mu_0, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dPhi2, F.dPhi2, P.BSize, cudaMemcpyHostToDevice);

  cudaMemcpy(f.p_p, F.p_p, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.px, F.px, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.py, F.py, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.pz, F.pz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.px_star, F.px_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.py_star, F.py_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.pz_star, F.pz_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.mu_x, F.mu_x, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.mu_y, F.mu_y, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.mu_z, F.mu_z, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.P2, F.P2, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtpx_star, F.dtpx_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtpy_star, F.dtpy_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dtpz_star, F.dtpz_star, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.OPx, F.OPx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.OPy, F.OPy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.OPz, F.OPz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.DPx, F.DPx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.DPy, F.DPy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.DPz, F.DPz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dxPx, F.dxPx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dyPx, F.dyPx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dzPx, F.dzPx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dxPy, F.dxPy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dyPy, F.dyPy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dzPy, F.dzPy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dxPz, F.dxPz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dyPz, F.dyPz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.dzPz, F.dzPz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigxx, F.Sigxx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigxy, F.Sigxy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigxz, F.Sigxz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigyx, F.Sigyx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigyy, F.Sigyy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigyz, F.Sigyz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigzx, F.Sigzx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigzy, F.Sigzy, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Sigzz, F.Sigzz, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Axx, F.Axx, P.BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(f.Axy, F.Axy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(f.Axz, F.Axz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(f.Ayy, F.Ayy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(f.Ayz, F.Ayz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(f.Azz, F.Azz, P.BSize, cudaMemcpyDeviceToHost);

  InputFile.close();
}

//=======================================================================
void ExpoConf(string str_t)
{
  ofstream ConfFile;
  int PrecData = 15;

  cudaMemcpy(F.vx, f.vx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.vy, f.vy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.vz, f.vz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.vx_star, f.vx_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.vy_star, f.vy_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.vz_star, f.vz_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtvx_star, f.dtvx_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtvy_star, f.dtvy_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtvz_star, f.dtvz_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.p_v, f.p_v, P.BSize, cudaMemcpyDeviceToHost);

  cudaMemcpy(F.px, f.px, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.py, f.py, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.pz, f.pz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.px_star, f.px_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.py_star, f.py_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.pz_star, f.pz_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtpx_star, f.dtpx_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtpy_star, f.dtpy_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtpz_star, f.dtpz_star, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.p_p, f.p_p, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.mu_x, f.mu_x, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.mu_y, f.mu_y, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.mu_z, f.mu_z, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.P2, f.P2, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.DPx, f.DPx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.DPy, f.DPy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.DPz, f.DPz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.OPx, f.OPx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.OPy, f.OPy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.OPz, f.OPz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dxPx, f.dxPx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dyPx, f.dyPx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dzPx, f.dzPx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dxPy, f.dxPy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dyPy, f.dyPy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dzPy, f.dzPy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dxPz, f.dxPz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dyPz, f.dyPz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dzPz, f.dzPz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigxx, f.Sigxx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigxy, f.Sigxy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigxz, f.Sigxz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigyx, f.Sigyx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigyy, f.Sigyy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigyz, f.Sigyz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigzx, f.Sigzx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigzy, f.Sigzy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Sigzz, f.Sigzz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Axx, f.Axx, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Axy, f.Axy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Axz, f.Axz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Ayy, f.Ayy, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Ayz, f.Ayz, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.Azz, f.Azz, P.BSize, cudaMemcpyDeviceToHost);

  cudaMemcpy(F.phi, f.phi, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.mu_0, f.mu_0, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dtphi, f.dtphi, P.BSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(F.dPhi2, f.dPhi2, P.BSize, cudaMemcpyDeviceToHost);
  

  if (P.ExpoData == -1)
  {

  }

  std::string ConfFileName = "data_" + to_string(gpu_node) + "/conf_" + str_t + ".dat";
  ConfFile.open(ConfFileName.c_str());
  int idx;
  // cout.precision(15);
  for (int i = 0; i < P.Ny; i++)
  {
    for (int j = 0; j < P.Nx; j++)
    {
      for (int k = 0; k < P.Nz; k++)
      {
      idx = (P.Nx + 2 * P.Nb)  * (P.Ny + 2 * P.Nb) * (k + P.Nb) + (P.Nx + 2 * P.Nb) * (i + P.Nb) + j + P.Nb;
      // idx = (P.Nx + 2 * P.Nb)  * (P.Ny + 2 * P.Nb) * k  + (P.Nx + 2 * P.Nb) * i + j;
        if (P.ExpoData == 1)
        {
          // ConfFile << fixed << setprecision(PrecData) << F.Qxx[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.Qxy[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.vx[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.vy[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.vz[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.px[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.py[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.pz[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.vx_star[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.vy_star[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.vz_star[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.mu_0[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.div_v_star[idx] << ' '
          ConfFile << fixed << setprecision(PrecData) << F.phi[idx] << endl;
          // ConfFile << fixed << setprecision(PrecData) << F.dtphi[idx] << ' ';
          // ConfFile << fixed << setprecision(PrecData) << F.p[idx] << endl;
          // add
          //cout <<"i=" << i << " j=" <<j << "gammav="<< F.gammav[idx] << endl;
          // ConfFile << fixed << setprecision(PrecData) << F.gammav[idx] << endl;
        }
        else if (P.ExpoData == -1)
        {
          ConfFile << fixed << setprecision(PrecData) << F.vx[idx] << ' ';
          ConfFile << fixed << setprecision(PrecData) << F.vy[idx] << ' ';
        }
      }
    }
  }
  ConfFile.close();
}

//=======================================================================
__global__ void getMaxX(Fields ff, real *dtVarMX)
{
  extern __shared__ real sdata[];
  int i = blockIdx.x;
  int j = threadIdx.x;
  // int idxA = (blockDim.x + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  // int idxB = (blockDim.x + 2 * p.Nb) * (i - p.Ny + p.Nb) + j + p.Nb;

  if (i < p.Ny)
  {
    // sdata[j] = abs(ff.dtQxx[idxA]);
  }
  else
  {
    // sdata[j] = abs(ff.dtvortex[idxB]);
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s >= 1; s = s / 2)
  {
    if (j < s)
    {
      if (sdata[j] < sdata[j + s])
      {
        sdata[j] = sdata[j + s];
      }
    }
    __syncthreads();
  }

  if (j == 0)
  {
    dtVarMX[i] = sdata[0];
  }
}

//=======================================================================
__global__ void getDt(real *dtVarMX, real *dtVarM, real *dt)
{
  extern __shared__ real sdata[];
  unsigned int i = blockIdx.x;
  unsigned int j = threadIdx.x;
  int idx = blockDim.x * i + j;

  // each thread loads one element from global to shared mem
  sdata[j] = dtVarMX[idx];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    if (j % (2 * s) == 0 && sdata[j] < sdata[j + s])
    {
      sdata[j] = sdata[j + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (j == 0)
  {
    dtVarM[0] = sdata[0];
    dt[0] = p.dtVarC / sdata[0];
    if (dt[0] > p.dtMax)
    {
      dt[0] = p.dtMax;
    }
  }
}

//=======================================================================
void initPoissonSolver()
{
  // Creating wavenumber array
  real kx, ky, kz;
  for (int i = 0; i < P.Ny; i++)
  {
    ky = 2 * Pi * i / (P.Ny * P.h + 0.0);
    if (i >= P.Ny / 2)
    {
      ky = 2 * Pi * (i - P.Ny) / (P.Ny * P.h + 0.0);
    }
    for (int j = 0; j < P.Nx; j++)
    {
      kx = 2 * Pi * j / (P.Nx * P.h + 0.0);
      if (j >= P.Nx / 2)
      {
        kx = 2 * Pi * (j - P.Nx) / (P.Nx * P.h + 0.0);
      }
      for (int k = 0; k < P.Nz; k++)
      {
        kz = 2 * Pi * k / (P.Nz * P.h + 0.0);
        if (k >= P.Nz / 2)
        {
          kz = 2 * Pi * (k - P.Nz) / (P.Nz * P.h + 0.0);
        }
        int idx = (P.Nx * P.Ny) * k + i * P.Nx + j;
        PS.k2[idx] = kx * kx + ky * ky + kz * kz;
      }
      // int idx = i * P.Nx + j;
      // PS.k2[idx] = kx * kx + ky * ky;
    }
  }
  PS.k2[0] = 1;

  cudaMemcpy(ps.k2, PS.k2, sizeof(real) * P.Nx * P.Ny * P.Nz, cudaMemcpyHostToDevice);
  cufftPlan3d(&plan, P.Ny, P.Nx, P.Nz, CUFFT_Z2Z);
}

//=======================================================================
void MemAlloc()
{
  // Allocate fields in host memory.
  F.vx = new real[P.GSize];
  F.vy = new real[P.GSize];
  F.vz = new real[P.GSize];
  F.vx_star = new real[P.GSize];
  F.vy_star = new real[P.GSize];
  F.vz_star = new real[P.GSize];
  F.dtvx_star = new real[P.GSize];
  F.dtvy_star = new real[P.GSize];
  F.dtvz_star = new real[P.GSize];
  F.p_v = new real[P.GSize];
  
  F.phi = new real[P.GSize];
  F.dtphi = new real[P.GSize];
  F.mu_0 = new real[P.GSize];
  F.dPhi2 = new real[P.GSize];

  F.px = new real[P.GSize];
  F.py = new real[P.GSize];
  F.pz = new real[P.GSize];
  F.px_star = new real[P.GSize];
  F.py_star = new real[P.GSize];
  F.pz_star = new real[P.GSize];
  F.dtpx_star = new real[P.GSize];
  F.dtpy_star = new real[P.GSize];
  F.dtpz_star = new real[P.GSize];
  F.P2 = new real[P.GSize];
  F.DPx = new real[P.GSize];
  F.DPy = new real[P.GSize];
  F.DPz = new real[P.GSize];
  F.OPx = new real[P.GSize];
  F.OPy = new real[P.GSize];
  F.OPz = new real[P.GSize];
  F.mu_x = new real[P.GSize];
  F.mu_y = new real[P.GSize];
  F.mu_z = new real[P.GSize];
  F.dxPx = new real[P.GSize];
  F.dyPx = new real[P.GSize];
  F.dzPx = new real[P.GSize];
  F.dxPy = new real[P.GSize];
  F.dyPy = new real[P.GSize];
  F.dzPy = new real[P.GSize];
  F.dxPz = new real[P.GSize];
  F.dyPz = new real[P.GSize];
  F.dzPz = new real[P.GSize];
  F.Sigxx = new real[P.GSize];
  F.Sigxy = new real[P.GSize];
  F.Sigxz = new real[P.GSize];
  F.Sigyx = new real[P.GSize];
  F.Sigyy = new real[P.GSize];
  F.Sigyz = new real[P.GSize];
  F.Sigzx = new real[P.GSize];
  F.Sigzy = new real[P.GSize];
  F.Sigzz = new real[P.GSize];
  F.Axx = new real[P.GSize];
  F.Axy = new real[P.GSize];
  F.Axz = new real[P.GSize];
  F.Ayy = new real[P.GSize];
  F.Ayz = new real[P.GSize];
  F.Azz = new real[P.GSize];
  F.p_p = new real[P.GSize];

  F.Fstream = new Complex[P.GSize];
  PRKC2.mu = new real[P.RKStage + 1];
  PRKC2.mu1 = new real[P.RKStage + 1];
  PRKC2.nu = new real[P.RKStage + 1];
  PRKC2.gamma1 = new real[P.RKStage + 1];
  PS.k2 = new real[P.Nx * P.Ny * P.Nz];

  // Allocate memory of fields in device.
  cudaMalloc((void **)&f.vx, P.BSize);
  cudaMalloc((void **)&f.vy, P.BSize);
  cudaMalloc((void **)&f.vz, P.BSize);
  cudaMalloc((void **)&f.vx_star, P.BSize);
  cudaMalloc((void **)&f.vy_star, P.BSize);
  cudaMalloc((void **)&f.vz_star, P.BSize);
  cudaMalloc((void **)&f.dtvx_star, P.BSize);
  cudaMalloc((void **)&f.dtvy_star, P.BSize);
  cudaMalloc((void **)&f.dtvz_star, P.BSize);
  cudaMalloc((void **)&f.p_v, P.BSize);

  cudaMalloc((void **)&f.phi, P.BSize);
  cudaMalloc((void **)&f.dtphi, P.BSize);
  cudaMalloc((void **)&f.mu_0, P.BSize);
  cudaMalloc((void **)&f.dPhi2, P.BSize);

  cudaMalloc((void **)&f.px, P.BSize);
  cudaMalloc((void **)&f.py, P.BSize);
  cudaMalloc((void **)&f.pz, P.BSize);
  cudaMalloc((void **)&f.px_star, P.BSize);
  cudaMalloc((void **)&f.py_star, P.BSize);
  cudaMalloc((void **)&f.pz_star, P.BSize);
  cudaMalloc((void **)&f.dtpx_star, P.BSize);
  cudaMalloc((void **)&f.dtpy_star, P.BSize);
  cudaMalloc((void **)&f.dtpz_star, P.BSize);
  cudaMalloc((void **)&f.P2, P.BSize);
  cudaMalloc((void **)&f.DPx, P.BSize);
  cudaMalloc((void **)&f.DPy, P.BSize);
  cudaMalloc((void **)&f.DPz, P.BSize);
  cudaMalloc((void **)&f.OPx, P.BSize);
  cudaMalloc((void **)&f.OPy, P.BSize);
  cudaMalloc((void **)&f.OPz, P.BSize);
  cudaMalloc((void **)&f.mu_x, P.BSize);
  cudaMalloc((void **)&f.mu_y, P.BSize);
  cudaMalloc((void **)&f.mu_z, P.BSize);
  cudaMalloc((void **)&f.dxPx, P.BSize);
  cudaMalloc((void **)&f.dyPx, P.BSize);
  cudaMalloc((void **)&f.dzPx, P.BSize);
  cudaMalloc((void **)&f.dxPy, P.BSize);
  cudaMalloc((void **)&f.dyPy, P.BSize);
  cudaMalloc((void **)&f.dzPy, P.BSize);
  cudaMalloc((void **)&f.dxPz, P.BSize);
  cudaMalloc((void **)&f.dyPz, P.BSize);
  cudaMalloc((void **)&f.dzPz, P.BSize);
  cudaMalloc((void **)&f.Sigxx, P.BSize);
  cudaMalloc((void **)&f.Sigxy, P.BSize);
  cudaMalloc((void **)&f.Sigxz, P.BSize);
  cudaMalloc((void **)&f.Sigyx, P.BSize);
  cudaMalloc((void **)&f.Sigyy, P.BSize);
  cudaMalloc((void **)&f.Sigyz, P.BSize);
  cudaMalloc((void **)&f.Sigzx, P.BSize);
  cudaMalloc((void **)&f.Sigzy, P.BSize);
  cudaMalloc((void **)&f.Sigzz, P.BSize);
  cudaMalloc((void **)&f.Axx, P.BSize);
  cudaMalloc((void **)&f.Axy, P.BSize);
  cudaMalloc((void **)&f.Axz, P.BSize);
  cudaMalloc((void **)&f.Ayy, P.BSize);
  cudaMalloc((void **)&f.Ayz, P.BSize);
  cudaMalloc((void **)&f.Azz, P.BSize);
  cudaMalloc((void **)&f.p_p, P.BSize);

  cudaMalloc((void **)&f.Fstream, sizeof(cufftDoubleComplex) * P.Nx * P.Ny * P.Nz);

  cudaMalloc((void **)&dtVarMX, 2 * P.Ny * sizeof(real));
  cudaMalloc((void **)&dtVarM, sizeof(real));
  cudaMalloc((void **)&t, sizeof(real));
  cudaMalloc((void **)&dt, sizeof(real));

  cudaMalloc((void **)&pRKC2.mu, (P.RKStage + 1) * sizeof(real));
  cudaMalloc((void **)&pRKC2.nu, (P.RKStage + 1) * sizeof(real));
  cudaMalloc((void **)&pRKC2.mu1, (P.RKStage + 1) * sizeof(real));
  cudaMalloc((void **)&pRKC2.gamma1, (P.RKStage + 1) * sizeof(real));
  cudaMalloc((void **)&ps.k2, (P.Nx * P.Ny * P.Nz) * sizeof(real));
}

//=======================================================================
void MemFree()
{
  // Free host memory
  delete[] F.vx;
  delete[] F.vy;
  delete[] F.vz;
  delete[] F.vx_star;
  delete[] F.vy_star;
  delete[] F.vz_star;
  delete[] F.dtvx_star;
  delete[] F.dtvy_star;
  delete[] F.dtvz_star;
  delete[] F.p_v;

  delete[] F.phi;
  delete[] F.dtphi;
  delete[] F.mu_0;
  delete[] F.dPhi2;

  delete[] F.px;
  delete[] F.py;
  delete[] F.pz;
  delete[] F.px_star;
  delete[] F.py_star;
  delete[] F.pz_star;
  delete[] F.dtpx_star;
  delete[] F.dtpy_star;
  delete[] F.dtpz_star;
  delete[] F.P2;
  delete[] F.DPx;
  delete[] F.DPy;
  delete[] F.DPz;
  delete[] F.OPx;
  delete[] F.OPy;
  delete[] F.OPz;
  delete[] F.mu_x;
  delete[] F.mu_y;
  delete[] F.mu_z;
  delete[] F.dxPx;
  delete[] F.dyPx;
  delete[] F.dzPx;
  delete[] F.dxPy;
  delete[] F.dyPy;
  delete[] F.dzPy;
  delete[] F.dxPz;
  delete[] F.dyPz;
  delete[] F.dzPz;
  delete[] F.Sigxx;
  delete[] F.Sigxy;
  delete[] F.Sigxz;
  delete[] F.Sigyx;
  delete[] F.Sigyy;
  delete[] F.Sigyz;
  delete[] F.Sigzx;
  delete[] F.Sigzy;
  delete[] F.Sigzz;
  delete[] F.Axx;
  delete[] F.Axy;
  delete[] F.Axz;
  delete[] F.Ayy;
  delete[] F.Ayz;
  delete[] F.Azz;
  delete[] F.p_p;

  delete[] F.Fstream;

  delete[] PRKC2.mu;
  delete[] PRKC2.nu;
  delete[] PRKC2.mu1;
  delete[] PRKC2.gamma1;
  delete[] PS.k2;

  // Free device memory
  cudaFree(f.vx);
  cudaFree(f.vy);
  cudaFree(f.vz);
  cudaFree(f.vx_star);
  cudaFree(f.vy_star);
  cudaFree(f.vz_star);
  cudaFree(f.dtvx_star);
  cudaFree(f.dtvy_star);
  cudaFree(f.dtvz_star);
  cudaFree(f.p_v);

  cudaFree(f.phi);
  cudaFree(f.dtphi);
  cudaFree(f.mu_0);
  cudaFree(f.dPhi2);

  cudaFree(f.px);
  cudaFree(f.py);
  cudaFree(f.pz);
  cudaFree(f.px_star);
  cudaFree(f.py_star);
  cudaFree(f.pz_star);
  cudaFree(f.dtpx_star);
  cudaFree(f.dtpy_star);
  cudaFree(f.dtpz_star);
  cudaFree(f.P2);
  cudaFree(f.DPx);
  cudaFree(f.DPy);
  cudaFree(f.DPz);
  cudaFree(f.OPx);
  cudaFree(f.OPy);
  cudaFree(f.OPz);
  cudaFree(f.mu_x);
  cudaFree(f.mu_y);
  cudaFree(f.mu_z);
  cudaFree(f.dxPx);
  cudaFree(f.dyPx);
  cudaFree(f.dzPx);
  cudaFree(f.dxPy);
  cudaFree(f.dyPy);
  cudaFree(f.dzPy);
  cudaFree(f.dxPz);
  cudaFree(f.dyPz);
  cudaFree(f.dzPz);
  cudaFree(f.Sigxx);
  cudaFree(f.Sigxy);
  cudaFree(f.Sigxz);
  cudaFree(f.Sigyx);
  cudaFree(f.Sigyy);
  cudaFree(f.Sigyz);
  cudaFree(f.Sigzx);
  cudaFree(f.Sigzy);
  cudaFree(f.Sigzz);
  cudaFree(f.Axx);
  cudaFree(f.Axy);
  cudaFree(f.Axz);
  cudaFree(f.Ayy);
  cudaFree(f.Ayz);
  cudaFree(f.Azz);
  cudaFree(f.p_p);

  cudaFree(f.Fstream);


  cudaFree(dtVarMX);
  cudaFree(dtVarM);
  cudaFree(pRKC2.mu);
  cudaFree(pRKC2.mu1);
  cudaFree(pRKC2.nu);
  cudaFree(pRKC2.gamma1);
  cudaFree(ps.k2);
}

//=======================================================================
__global__ void BounPeriU(real *u)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb) * (k + p.Nb) + (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int idx1;
  int dj = p.Nx + 2 * p.Nb;
  int dk = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb);

  if (k < p.Nb)
  {
    // Bottom face
    idx1 = idx + p.Nz * dk;
    u[idx1] = u[idx];
  }
  else if (k > p.Nz - 1 - p.Nb)
  {
    // Top face
    idx1 = idx - p.Nz * dk;
    u[idx1] = u[idx];
  }

  if (j < p.Nb)
  {
    // Left face
    idx1 = idx + p.Nx;
    u[idx1] = u[idx];
    
    if (k < p.Nb)
    {
      // Bottom left corner
      idx1 = idx + dk * p.Nz + p.Nx;
      u[idx1] = u[idx];
    }
    else if (k > p.Nz - 1 - p.Nb)
    {
      // Top left corner
      idx1 = idx - dk * p.Nz + p.Nx;
      u[idx1] = u[idx];
    }

    if (i < p.Nb)
    {
      // Front left corner
      idx1 = idx + dj * p.Ny + p.Nx;
      u[idx1] = u[idx];
    }
    else if (i > p.Ny - 1 - p.Nb)
    {
      // Back left corner
      idx1 = idx - dj * p.Ny + p.Nx;
      u[idx1] = u[idx];
    }
  }
  else if (j > p.Nx - 1 - p.Nb)
  {
    // Right face
    idx1 = idx - p.Nx;
    u[idx1] = u[idx];
    
    if (k < p.Nb)
    {
      // Bottom right corner
      idx1 = idx + dk * p.Nz - p.Nx;
      u[idx1] = u[idx];
    }
    else if (k > p.Nz - 1 - p.Nb)
    {
      // Top right corner
      idx1 = idx - dk * p.Nz - p.Nx;
      u[idx1] = u[idx];
    }

    if (i < p.Nb)
    {
      // Front right corner
      idx1 = idx + dj * p.Ny - p.Nx;
      u[idx1] = u[idx];
    }
    else if (i > p.Ny - 1 - p.Nb)
    {
      // Back right corner
      idx1 = idx - dj * p.Ny - p.Nx;
      u[idx1] = u[idx];
    }
  }

  if (i < p.Nb)
  {

    // Front face
    idx1 = idx + dj * p.Ny;
    u[idx1] = u[idx];

    if (k < p.Nb)
    {
      // Front bottom corner
      idx1 = idx + dj * p.Ny + dk * p.Nz;
      u[idx1] = u[idx];

      if (j < p.Nb)
      {
        // Front bottom left corner
        idx1 = idx + dj * p.Ny + dk * p.Nz + p.Nx;
        u[idx1] = u[idx];
      }
      else if (j > p.Nx - 1 - p.Nb)
      {
        // Front bottom right corner
        idx1 = idx + dj * p.Ny + dk * p.Nz - p.Nx;
        u[idx1] = u[idx];
      }

    }
    else if (k > p.Nz - 1 - p.Nb)
    {
      // Front top corner
      idx1 = idx + dj * p.Ny - dk * p.Nz;
      u[idx1] = u[idx];

      if (j < p.Nb)
      {
        // Front top left corner
        idx1 = idx + dj * p.Ny - dk * p.Nz + p.Nx;
        u[idx1] = u[idx];
      }
      else if (j > p.Nx - 1 - p.Nb)
      {
        // Front top right corner
        idx1 = idx + dj * p.Ny - dk * p.Nz - p.Nx;
        u[idx1] = u[idx];
      }
    }
  }
  else if (i > p.Ny - 1 - p.Nb)
  {

    // Back face
    idx1 = idx - dj * p.Ny;
    u[idx1] = u[idx];

    if (k < p.Nb)
    {
      // Back bottom corner
      idx1 = idx - dj * p.Ny + dk * p.Nz;
      u[idx1] = u[idx];

      if (j < p.Nb)
      {
        // Back bottom left corner
        idx1 = idx - dj * p.Ny + dk * p.Nz + p.Nx;
        u[idx1] = u[idx];
      }
      else if (j > p.Nx - 1 - p.Nb)
      {
        // Back bottom right corner
        idx1 = idx - dj * p.Ny + dk * p.Nz - p.Nx;
        u[idx1] = u[idx];
      }

    }
    else if (k > p.Nz - 1 - p.Nb)
    {
      // Back top corner
      idx1 = idx - dj * p.Ny - dk * p.Nz;
      u[idx1] = u[idx];

      if (j < p.Nb)
      {
        // Back top left corner
        idx1 = idx - dj * p.Ny - dk * p.Nz + p.Nx;
        u[idx1] = u[idx];
      }
      else if (j > p.Nx - 1 - p.Nb)
      {
        // Back top right corner
        idx1 = idx - dj * p.Ny - dk * p.Nz - p.Nx;
        u[idx1] = u[idx];
      }
    }
  }



}

//=======================================================================
__global__ void BounPeriF(Fields ff)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  int k = blockIdx.y;
  int idx = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb) * (k + p.Nb) + (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int idx1;
  int dj = p.Nx + 2 * p.Nb;
  int dk = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb);


  if (k < p.Nb)
  {
    // Bottom face
    idx1 = idx + p.Nz * dk;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];


    // u[idx1] = u[idx];
  }
  else if (k > p.Nz - 1 - p.Nb)
  {
    // Top face
    idx1 = idx - p.Nz * dk;
    idx1 = idx + p.Nz * dk;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  if (j < p.Nb)
  {
    // Left face
    idx1 = idx + p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
    // u[idx1] = u[idx];
  }
  else if (j > p.Nx - 1 - p.Nb)
  {
    // Right face
    idx1 = idx - p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
    // u[idx1] = u[idx];
  }

  if (i < p.Nb)
  {
    // Front face
    idx1 = idx + dj * p.Ny;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
    // u[idx1] = u[idx];
  }
  else if (i > p.Ny - 1 - p.Nb)
  {
    // Back face
    idx1 = idx - dj * p.Ny;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
    // u[idx1] = u[idx];
  }

  if (j < p.Nb && i < p.Nb)
  {
    // Front left corner
    idx1 = idx + dj * p.Ny + p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }  
  else if (j > p.Nx - 1 - p.Nb && i < p.Nb)
  {
    // Front right corner
    idx1 = idx + dj * p.Ny - p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j < p.Nb && i > p.Ny - 1 - p.Nb)
  {
    // Back left corner
    idx1 = idx - dj * p.Ny + p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && i > p.Ny - 1 - p.Nb)
  {
    // Back right corner
    idx1 = idx - dj * p.Ny - p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }

  if (j < p.Nb && k < p.Nb)
  {
    // Bottom left corner
    idx1 = idx + dk * p.Nz + p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && k < p.Nb)
  {
    // Bottom right corner
    idx1 = idx + dk * p.Nz - p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j < p.Nb && k > p.Nz - 1 - p.Nb)
  {
    // Top left corner
    idx1 = idx - dk * p.Nz + p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && k > p.Nz - 1 - p.Nb)
  {
    // Top right corner
    idx1 = idx - dk * p.Nz - p.Nx;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }

  if (j < p.Nb && i < p.Nb && k < p.Nb)
  {
    // Front bottom left corner
    idx1 = idx + dj * p.Ny + p.Nx + dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && i < p.Nb && k < p.Nb)
  {
    // Front bottom right corner
    idx1 = idx + dj * p.Ny - p.Nx + dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j < p.Nb && i > p.Ny - 1 - p.Nb && k < p.Nb)
  {
    // Front top left corner
    idx1 = idx + dj * p.Ny + p.Nx - dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && i > p.Ny - 1 - p.Nb && k < p.Nb)
  {
    // Front top right corner
    idx1 = idx + dj * p.Ny - p.Nx - dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j < p.Nb && i < p.Nb && k > p.Nz - 1 - p.Nb)
  {
    // Back bottom left corner
    idx1 = idx + dj * p.Ny + p.Nx - dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && i < p.Nb && k > p.Nz - 1 - p.Nb)
  {
    // Back bottom right corner
    idx1 = idx + dj * p.Ny - p.Nx - dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j < p.Nb && i > p.Ny - 1 - p.Nb && k > p.Nz - 1 - p.Nb)
  {
    // Back top left corner
    idx1 = idx + dj * p.Ny + p.Nx - dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }
  else if (j > p.Nx - 1 - p.Nb && i > p.Ny - 1 - p.Nb && k > p.Nz - 1 - p.Nb)
  {
    // Back top right corner
    idx1 = idx + dj * p.Ny - p.Nx - dk * p.Nz;
    ff.vx[idx1] = ff.vx[idx];
    ff.vy[idx1] = ff.vy[idx];
    ff.vz[idx1] = ff.vz[idx];
    ff.vx_star[idx1] = ff.vx_star[idx];
    ff.vy_star[idx1] = ff.vy_star[idx];
    ff.vz_star[idx1] = ff.vz_star[idx];
    ff.dtvx_star[idx1] = ff.dtvx_star[idx];
    ff.dtvy_star[idx1] = ff.dtvy_star[idx];
    ff.dtvz_star[idx1] = ff.dtvz_star[idx];
    ff.p_v[idx1] = ff.p_v[idx];

    ff.phi[idx1] = ff.phi[idx];
    ff.dtphi[idx1] = ff.dtphi[idx];
    ff.mu_0[idx1] = ff.mu_0[idx];
    ff.dPhi2[idx1] = ff.dPhi2[idx];

    ff.px[idx1] = ff.px[idx];
    ff.py[idx1] = ff.py[idx];
    ff.pz[idx1] = ff.pz[idx];
    ff.px_star[idx1] = ff.px_star[idx];
    ff.py_star[idx1] = ff.py_star[idx];
    ff.pz_star[idx1] = ff.pz_star[idx];
    ff.dtpx_star[idx1] = ff.dtpx_star[idx]; 
    ff.dtpy_star[idx1] = ff.dtpy_star[idx];
    ff.dtpz_star[idx1] = ff.dtpz_star[idx];
    ff.P2[idx1] = ff.P2[idx];
    ff.DPx[idx1] = ff.DPx[idx];
    ff.DPy[idx1] = ff.DPy[idx];
    ff.DPz[idx1] = ff.DPz[idx];
    ff.OPx[idx1] = ff.OPx[idx];
    ff.OPy[idx1] = ff.OPy[idx];
    ff.OPz[idx1] = ff.OPz[idx];
    ff.mu_x[idx1] = ff.mu_x[idx];
    ff.mu_y[idx1] = ff.mu_y[idx];
    ff.mu_z[idx1] = ff.mu_z[idx];
    ff.dxPx[idx1] = ff.dxPx[idx];
    ff.dyPx[idx1] = ff.dyPx[idx];
    ff.dzPx[idx1] = ff.dzPx[idx];
    ff.dxPy[idx1] = ff.dxPy[idx];
    ff.dyPy[idx1] = ff.dyPy[idx];
    ff.dzPy[idx1] = ff.dzPy[idx];
    ff.dxPz[idx1] = ff.dxPz[idx];
    ff.dyPz[idx1] = ff.dyPz[idx];
    ff.dzPz[idx1] = ff.dzPz[idx];
    ff.Sigxx[idx1] = ff.Sigxx[idx];
    ff.Sigxy[idx1] = ff.Sigxy[idx];
    ff.Sigxz[idx1] = ff.Sigxz[idx];
    ff.Sigyx[idx1] = ff.Sigyx[idx];
    ff.Sigyy[idx1] = ff.Sigyy[idx];
    ff.Sigyz[idx1] = ff.Sigyz[idx];
    ff.Sigzx[idx1] = ff.Sigzx[idx];
    ff.Sigzy[idx1] = ff.Sigzy[idx];
    ff.Sigzz[idx1] = ff.Sigzz[idx];
    ff.Axx[idx1] = ff.Axx[idx];
    ff.Axy[idx1] = ff.Axy[idx];
    ff.Axz[idx1] = ff.Axz[idx];
    ff.Ayy[idx1] = ff.Ayy[idx];
    ff.Ayz[idx1] = ff.Ayz[idx];
    ff.Azz[idx1] = ff.Azz[idx];
    ff.p_p[idx1] = ff.p_p[idx];
  }  
  
      
}



//=======================================================================
__device__ real d1xO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (12.0 * p.h) * ((u[idx + di + dj] - u[idx + di - dj]) + 4 * (u[idx + dj] - u[idx - dj]) + (u[idx - di + dj] - u[idx - di - dj]));
}

//=======================================================================
__device__ real d1yO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (12.0 * p.h) * ((u[idx + di + dj] - u[idx - di + dj]) + 4 * (u[idx + di] - u[idx - di]) + (u[idx + di - dj] - u[idx - di - dj]));
}

//=======================================================================
__device__ real d2xO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (12.0 * p.h * p.h) * ((u[idx + di + dj] - 2 * u[idx + di] + u[idx + di - dj]) + 10 * (u[idx + dj] - 2 * u[idx] + u[idx - dj]) + (u[idx - di + dj] - 2 * u[idx - di] + u[idx - di - dj]));
}

//=======================================================================
__device__ real d2yO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (12.0 * p.h * p.h) * ((u[idx + di + dj] - 2 * u[idx + dj] + u[idx - di + dj]) + 10 * (u[idx + di] - 2 * u[idx] + u[idx - di]) + (u[idx + di - dj] - 2 * u[idx - dj] + u[idx - di - dj]));
}

//===========================================================================
__device__ real d1x1yO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (4.0 * p.h * p.h) * (u[idx + di + dj] - u[idx - di + dj] - u[idx + di - dj] + u[idx - di - dj]);
}

// ======================================================================
__device__ real LaplO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 2) * (-10.0 / 3.0 * u[idx] + 2.0 / 3.0 * (u[idx + di] + u[idx - di] + u[idx + dj] + u[idx - dj]) + 1.0 / 6.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]));
}

// ======================================================================
__device__ real BiLaO2I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 4) * (12 * u[idx] - 10.0 / 3.0 * (u[idx + di] + u[idx - di] + u[idx + dj] + u[idx - dj]) - 2.0 / 3.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]) + 1.0 / 3.0 * (u[idx + 2 * di] + u[idx - 2 * di] + u[idx + 2 * dj] + u[idx - 2 * dj]) + 1.0 / 3.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] + u[idx + di - 2 * dj] + u[idx - di - 2 * dj] + u[idx + 2 * di + dj] + u[idx - 2 * di + dj] + u[idx + 2 * di - dj] + u[idx - 2 * di - dj]));
}

//=======================================================================
__device__ real d1xO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / p.h * (13.0 / 30.0 * (u[idx + dj] - u[idx - dj]) + 2.0 / 15.0 * (u[idx + di + dj] + u[idx - di + dj] - u[idx + di - dj] - u[idx - di - dj]) - 1.0 / 60.0 * (u[idx + 2 * dj] - u[idx - 2 * dj]) - 1.0 / 60.0 * (u[idx + 2 * di + dj] + u[idx - 2 * di + dj] - u[idx + 2 * di - dj] - u[idx - 2 * di - dj]) - 1.0 / 30.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] - u[idx + di - 2 * dj] - u[idx - di - 2 * dj]));
}

//=======================================================================
__device__ real d1yO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = p.Nx + 2 * p.Nb;
  int di = 1;
  return 1.0 / p.h * (13.0 / 30.0 * (u[idx + dj] - u[idx - dj]) + 2.0 / 15.0 * (u[idx + di + dj] + u[idx - di + dj] - u[idx + di - dj] - u[idx - di - dj]) - 1.0 / 60.0 * (u[idx + 2 * dj] - u[idx - 2 * dj]) - 1.0 / 60.0 * (u[idx + 2 * di + dj] + u[idx - 2 * di + dj] - u[idx + 2 * di - dj] - u[idx - 2 * di - dj]) - 1.0 / 30.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] - u[idx + di - 2 * dj] - u[idx - di - 2 * dj]));
}

//=======================================================================
__device__ real d2xO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 2) * (5.0 * u[idx] - 164.0 / 45.0 * (u[idx + dj] + u[idx - dj]) + 103.0 / 90.0 * (u[idx + 2 * dj] + u[idx - 2 * dj]) - 223.0 / 45.0 * (u[idx + di] + u[idx - di]) + 148.0 / 45.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]) - 73.0 / 90.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] + u[idx + di - 2 * dj] + u[idx - di - 2 * dj]) + 217.0 / 180.0 * (u[idx + 2 * di] + u[idx - 2 * di]) - 4.0 / 5.0 * (+u[idx + 2 * di + dj] + u[idx - 2 * di + dj] + u[idx + 2 * di - dj] + u[idx - 2 * di - dj]) + 71.0 / 360.0 * (+u[idx + 2 * di + 2 * dj] + u[idx - 2 * di + 2 * dj] + u[idx + 2 * di - 2 * dj] + u[idx - 2 * di - 2 * dj]));
}

//=======================================================================
__device__ real d2yO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int di = 1;
  int dj = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 2) * (5.0 * u[idx] - 164.0 / 45.0 * (u[idx + dj] + u[idx - dj]) + 103.0 / 90.0 * (u[idx + 2 * dj] + u[idx - 2 * dj]) - 223.0 / 45.0 * (u[idx + di] + u[idx - di]) + 148.0 / 45.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]) - 73.0 / 90.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] + u[idx + di - 2 * dj] + u[idx - di - 2 * dj]) + 217.0 / 180.0 * (u[idx + 2 * di] + u[idx - 2 * di]) - 4.0 / 5.0 * (+u[idx + 2 * di + dj] + u[idx - 2 * di + dj] + u[idx + 2 * di - dj] + u[idx - 2 * di - dj]) + 71.0 / 360.0 * (+u[idx + 2 * di + 2 * dj] + u[idx - 2 * di + 2 * dj] + u[idx + 2 * di - 2 * dj] + u[idx - 2 * di - 2 * dj]));
}

//=======================================================================
__device__ real d1x1yO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (p.h * p.h) * (17.0 / 45.0 * (u[idx - di - dj] + u[idx + di + dj] - u[idx + di - dj] - u[idx - di + dj]) - 1.0 / 45.0 * (u[idx + 2 * di + dj] + u[idx + di + 2 * dj] + u[idx - 2 * di - dj] + u[idx - di - 2 * dj] - u[idx - di + 2 * dj] - u[idx - 2 * di + dj] - u[idx + 2 * di - dj] - u[idx + di - 2 * dj]) - 7.0 / 720.0 * (u[idx - 2 * di - 2 * dj] + u[idx + 2 * di + 2 * dj] - u[idx - 2 * di + 2 * dj] - u[idx + 2 * di - 2 * dj]));
}

//======================================================================
__device__ real LaplO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 2) * (-21.0 / 5.0 * u[idx] + 13.0 / 15.0 * (u[idx + di] + u[idx - di] + u[idx + dj] + u[idx - dj]) + 4.0 / 15.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]) - 1.0 / 60.0 * (u[idx + 2 * di] + u[idx - 2 * di] + u[idx + 2 * dj] + u[idx - 2 * dj]) - 1.0 / 30.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] + u[idx + di - 2 * dj] + u[idx - di - 2 * dj] + u[idx + 2 * di + dj] + u[idx - 2 * di + dj] + u[idx + 2 * di - dj] + u[idx - 2 * di - dj]));
}

//=======================================================================
__device__ real BiLaO4I(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 4) * (779.0 / 45.0 * u[idx] - 191.0 / 45.0 * (u[idx + di] + u[idx - di] + u[idx + dj] + u[idx - dj]) - 187.0 / 90.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]) + 7.0 / 30.0 * (u[idx + 2 * di] + u[idx - 2 * di] + u[idx + 2 * dj] + u[idx - 2 * dj]) + 47.0 / 45.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] + u[idx + di - 2 * dj] + u[idx - di - 2 * dj] + u[idx + 2 * di + dj] + u[idx - 2 * di + dj] + u[idx + 2 * di - dj] + u[idx - 2 * di - dj]) - 29.0 / 180.0 * (u[idx + 2 * di + 2 * dj] + u[idx - 2 * di + 2 * dj] + u[idx + 2 * di - 2 * dj] + u[idx - 2 * di - 2 * dj]) + 1.0 / 45.0 * (u[idx + 3 * di] + u[idx - 3 * di] + u[idx + 3 * dj] + u[idx - 3 * dj]) - 17.0 / 180.0 * (u[idx + di + 3 * dj] + u[idx - di + 3 * dj] + u[idx + di - 3 * dj] + u[idx - di - 3 * dj] + u[idx + 3 * di + dj] + u[idx - 3 * di + dj] + u[idx + 3 * di - dj] + u[idx - 3 * di - dj]));
}

//=======================================================================
__device__ real d1xO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int didx = 1;
  return 1.0 / (12.0 * p.h) * (-u[idx + 2 * didx] + 8 * u[idx + didx] - 8 * u[idx - didx] + u[idx - 2 * didx]);
}

//=======================================================================
__device__ real d1yO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int didx = p.Nx + 2 * p.Nb;
  return 1.0 / (12.0 * p.h) * (-u[idx + 2 * didx] + 8 * u[idx + didx] - 8 * u[idx - didx] + u[idx - 2 * didx]);
}

//=======================================================================
__device__ real d2xO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int didx = 1;
  return 1.0 / (12.0 * p.h * p.h) * (-u[idx + 2 * didx] + 16 * u[idx + didx] - 30 * u[idx] + 16 * u[idx - didx] - u[idx - 2 * didx]);
}

//=======================================================================
__device__ real d2yO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int didx = p.Nx + 2 * p.Nb;
  return 1.0 / (12.0 * p.h * p.h) * (-u[idx + 2 * didx] + 16 * u[idx + didx] - 30 * u[idx] + 16 * u[idx - didx] - u[idx - 2 * didx]);
}

//=======================================================================
__device__ real d1x1yO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (144.0 * p.h * p.h) * (64 * (u[idx - di - dj] + u[idx + di + dj] - u[idx + di - dj] - u[idx - di + dj]) - 8 * (u[idx + 2 * di + dj] + u[idx + di + 2 * dj] + u[idx - 2 * di - dj] + u[idx - di - 2 * dj] - u[idx - di + 2 * dj] - u[idx - 2 * di + dj] - u[idx + 2 * di - dj] - u[idx + di - 2 * dj]) + (u[idx - 2 * di - 2 * dj] + u[idx + 2 * di + 2 * dj] - u[idx - 2 * di + 2 * dj] - u[idx + 2 * di - 2 * dj]));
}

//=======================================================================
__device__ real LaplO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (p.h * p.h) * (-5 * u[idx] + 4.0 / 3.0 * (u[idx + di] + u[idx - di] + u[idx + dj] + u[idx - dj]) - 1.0 / 12.0 * (u[idx + 2 * di] + u[idx - 2 * di] + u[idx + 2 * dj] + u[idx - 2 * dj]));
}

//=======================================================================
__device__ real BiLaO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / pow(p.h, 4) * (92.0 / 3.0 * u[idx] - 77.0 / 6.0 * (u[idx + di] + u[idx - di] + u[idx + dj] + u[idx - dj]) + 10.0 / 3.0 * (u[idx + di + dj] + u[idx - di + dj] + u[idx + di - dj] + u[idx - di - dj]) + 7.0 / 3.0 * (u[idx + 2 * di] + u[idx - 2 * di] + u[idx + 2 * dj] + u[idx - 2 * dj]) - 1.0 / 6.0 * (u[idx + di + 2 * dj] + u[idx - di + 2 * dj] + u[idx + di - 2 * dj] + u[idx - di - 2 * dj] + u[idx + 2 * di + dj] + u[idx - 2 * di + dj] + u[idx + 2 * di - dj] + u[idx - 2 * di - dj]) - 1.0 / 6.0 * (u[idx + 3 * di] + u[idx - 3 * di] + u[idx + 3 * dj] + u[idx - 3 * dj]));
}

//=======================================================================
__device__ real d4xO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int didx = 1;
  return 1.0 / (6.0 * pow(p.h, 4)) * (-u[idx + 3 * didx] + 12 * u[idx + 2 * didx] - 39 * u[idx + didx] + 56 * u[idx] - 39 * u[idx - didx] + 12 * u[idx - 2 * didx] - u[idx - 3 * didx]);
}

//=======================================================================
__device__ real d4yO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int didx = p.Nx + 2 * p.Nb;
  return 1.0 / (6.0 * pow(p.h, 4)) * (-u[idx + 3 * didx] + 12 * u[idx + 2 * didx] - 39 * u[idx + didx] + 56 * u[idx] - 39 * u[idx - didx] + 12 * u[idx - 2 * didx] - u[idx - 3 * didx]);
}

//=======================================================================
__device__ real d2x2yO4(real *u, int i, int j)
{
  int idx = (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
  int dj = 1;
  int di = p.Nx + 2 * p.Nb;
  return 1.0 / (16.0 * pow(p.h, 4)) * (u[idx + 2 * dj + 2 * di] - 2 * u[idx + 2 * dj] + u[idx - 2 * di + 2 * dj] - 2 * u[idx + 2 * di] + 4 * u[idx] - 2 * u[idx - 2 * di] + u[idx + 2 * di - 2 * dj] - 2 * u[idx - 2 * dj] + u[idx - 2 * di - 2 * dj]);
}

// --------------------------------------------------------
__device__ real d1xCO2I3D(real * f, int i, int j, int k) {
   
    int idx = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb) * (k + p.Nb) + (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
    int dj = 1;
    int di = p.Nx + 2 * p.Nb;
    int dk = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb);

    return 1.0 / (2.0 * p.h) * (
              16.0 / 36 * ( f[idx + dj] - f[idx - dj] )
              + 4.0 / 36 * ( f[idx + dj + dk] + f[idx + dj - dk] + f[idx + dj + di] + f[idx + dj - di]
              - f[idx - dj + dk] - f[idx - dj - dk] - f[idx - dj + di] - f[idx - dj - di] )
              + 1.0 / 36 * ( f[idx + dj + di + dk] + f[idx + dj - di + dk] + f[idx + dj + di - dk] + f[idx + dj - di - dk]
              - f[idx - dj + di + dk] - f[idx - dj - di + dk] - f[idx - dj + di - dk] - f[idx - dj - di - dk] )
              ); 
}

// --------------------------------------------------------
__device__ real d1yCO2I3D(real * f, int i, int j, int k) {
   
    int idx = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb) * (k + p.Nb) + (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
    int dj = 1;
    int di = p.Nx + 2 * p.Nb;
    int dk = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb);
    
    return 1.0 / (2.0 * p.h) * (
              16.0 / 36 * ( f[idx + di] - f[idx - di] )
              + 4.0 / 36 * ( f[idx + di + dk] + f[idx + di - dk] + f[idx + di + dj] + f[idx + di - dj]
              - f[idx - di + dk] - f[idx - di - dk] - f[idx - di + dj] - f[idx - di - dj] )
              + 1.0 / 36 * ( f[idx + di + dj + dk] + f[idx + di - dj + dk] + f[idx + di + dj - dk] + f[idx + di - dj - dk]
              - f[idx - di + dj + dk] - f[idx - di - dj + dk] - f[idx - di + dj - dk] - f[idx - di - dj - dk] )
              );
}

// --------------------------------------------------------
__device__ real d1zCO2I3D(real * f, int i, int j, int k) {
   
    int idx = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb) * (k + p.Nb) + (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
    int dj = 1;
    int di = p.Nx + 2 * p.Nb;
    int dk = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb);
    
    return 1.0 / (2.0 * p.h) * (
              16.0 / 36 * ( f[idx + dk] - f[idx - dk] )
              + 4.0 / 36 * ( f[idx + dk + dj] + f[idx + dk - dj] + f[idx + dk + di] + f[idx + dk - di]
              - f[idx - dk + dj] - f[idx - dk - dj] - f[idx - dk + di] - f[idx - dk - di] )
              + 1.0 / 36 * ( f[idx + dk + di + dj] + f[idx + dk - di + dj] + f[idx + dk + di - dj] + f[idx + dk -di - dj]
              - f[idx - dk + di + dj] - f[idx - dk - di + dj] - f[idx - dk + di - dj] - f[idx - dk - di - dj] )
              );
}

// -------------------------------------------------------
__device__ real laplaceCO2I3D(real * f, int i, int j, int k) {
     
    int idx = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb) * (k + p.Nb) + (p.Nx + 2 * p.Nb) * (i + p.Nb) + j + p.Nb;
    int dj = 1;
    int di = p.Nx + 2 * p.Nb;
    int dk = (p.Nx + 2 * p.Nb) * (p.Ny + 2 * p.Nb);

    return 1.0 / (12 * p.h * p.h ) * ( f[idx + di + dj - dk] + f[idx - di + dj - dk] + f[idx + di - dj - dk] + f[idx - di - dj - dk]
            + f[idx + di + dj + dk] + f[idx - di + dj + dk] + f[idx + di - dj + dk] + f[idx - di - dj + dk] )
            + 2.0 / (3 * p.h * p.h) * ( f[idx - dk] + f[idx + dk]
            + f[idx - dj] + f[idx + dj] + f[idx - di] + f[idx + di] )
            -14.0 / (3 * p.h * p.h) * f[idx];
}

//=======================================================================
void initRandSeed()
{
  // Initialize random seed in device.
  curandState_t *states;
  cudaMalloc((void **)&states, P.GSize * sizeof(curandState_t));
  initRandSeedDevi<<<P.Ny, P.Nx>>>(time(NULL), states);
}

//=======================================================================
__global__ void initRandSeedDevi(unsigned int seed, curandState_t *states)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, index, 0, &states[index]);
}

//=======================================================================
void getPRKC2()
{
  real s = P.RKStage;
  real *b;
  b = new real[P.RKStage + 1];
  real epslon = 2.0 / 13.0;
  real w0 = 1 + epslon / (s * s);
  real w01 = w0 * w0 - 1;
  real acw0 = acosh(w0);
  real w1 = sinh(s * acw0) * w01;
  w1 = w1 / (s * sqrt(w01) * cosh(s * acw0) - w0 * sinh(s * acw0));
  b[0] = (1.0 / tanh(2 * acw0) - w0 * 0.5 / sqrt(w01)) / sinh(2 * acw0);
  b[1] = b[0];
  b[2] = b[0];
  PRKC2.mu1[1] = b[1] * w1;

  for (int i = 2; i < s + 0.1; i++)
  {
    b[i] = (1.0 / tanh(i * acw0) - w0 / (i + 0.0) / sqrt(w01)) / sinh(i * acw0);
    PRKC2.mu[i] = 2 * b[i] * w0 / b[i - 1];
    PRKC2.nu[i] = -b[i] / b[i - 2];
    PRKC2.mu1[i] = 2 * b[i] * w1 / b[i - 1];
    PRKC2.gamma1[i] = (b[i - 1] * cosh((i - 1) * acw0) - 1) * PRKC2.mu1[i];
  }
}

//=======================================================================
void ShowProgress()
{
  // Print progress.
  progress = (T - P.T0) / (P.Ts - P.T0);
  int barWidth = 50;
  clock_t tNow = clock();
  double tUsed = double(tNow - tStart) / CLOCKS_PER_SEC;
  std::cout << "Progress: ";
  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i)
  {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %";
  if (T == 0)
  {
    std::cout << "\r";
  }
  else
  {
    std::cout << ".  " << floor(tUsed / progress * (1 - progress)) << "s remains.\r";
    // std::cout << ".  " << floor(tUsed/progress*(1-progress)) << "s, "<<Dt<<".\r";
  }
  std::cout.flush();
}

//=======================================================================
