/*
 
 trajectory.h
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
*/

#ifndef __trajectory_h__
#define __trajectory_h__

#include <complex>
#include <fftw3.h>
#include <vector>

#ifndef LIGHTWEIGHT
#define LIGHTWEIGHT 0
#endif

#define OPT_EVOL 1 //turns on/off check to see if jumps are necessary and potential rMax cutoff
#define FASTFORWARD 1  //turns on/off wavefunction "fast forward" (use prior Heff evolution up to first jump) 

typedef std::complex<double> dcomp;

extern double L,m,dt,dx,dk,alpha,T0,Tf,t0,tmed,kappa,gam,initWidth,rMax,mdfac;
extern int num,mem_size_dcomp,maxSteps,snapFreq,snapPts,potential,initType,initN,initL,initC,nmed,nTrajectories,projType,stepper,NLO;
extern int doJumps,nBasis,nThreads,derivType,temperatureEvolution,dirnameWithSeed,maxJumps,saveWavefunctions,outputSummaryFile;
extern long unsigned int randomseed;
extern dcomp *basisFunctions;
extern dcomp *wfnc,*wfncSave,*wfncInit;
extern std::ofstream outputFile;
extern std::string temperatureFile, basisFunctionsFile, outfileID;
extern int cstate,lval;
extern dcomp *spaceKernel,*momKernel;
extern double firstRandomNumber;

extern double *singletOverlaps0;
extern double *singletOverlaps;
extern std::vector<std::string> metadata;

extern double *T;

const double HBARC = 0.197326938; // GeV fm

const double TC = 0.15; // Tc in GeV

const int ABELIAN = -1;
const int SINGLET = 0;
const int OCTET = 1;

void initializeEvolution();
double evolveWavefunction(int nStart, int nSteps);
void evolveWavefunctionWithJumps(int nStart, double rmin);
void evolveWavefunctionWithJumpsCN(int nStart, double rmin);
void finalizeEvolution();
void saveNoJumpsSingletOverlaps();


// used to record wfnc from classical evolution
class SnapShot {
  private:
    std::vector<dcomp> data;

  public:
    
    double norm;
    int timeIdx;
    
    // constructor
    SnapShot() {
    }
    
    // destructor
    ~SnapShot() {
        data.resize(0);
    }
    
    void setData(dcomp* in) {
        data.resize(num);
        for (int i=0;i<num;i++)  data[i] = in[i];
    }
    
    void getData(dcomp* out) {
        for (int i=0;i<data.size();i++)  out[i] = data[i];
    }
};

#endif /* __trajectory_h__ */
