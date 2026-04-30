/*
 
 trajectory.cpp
 
 Copyright (c) Michael Strickland, Ajaharul Islam, and Anurag Tiwari
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdio.h>
#include <limits>
#include <gsl/gsl_sf_laguerre.h>

using namespace std;

#include "trajectory.h"
#include "wavefunction.h"
#include "outputroutines.h"
#include "interpolator.h"

#if LIGHTWEIGHT == 0
#include "eigensolver.h"
#endif

// =================================================================================
// Default values for all params; accessible by anything that includes trajectory.h
// These are overridden by the contents of input/params.txt
// =================================================================================

// number of Quantum trajectories
int nTrajectories = 1;

// number of grid points
int num = 1024;

// length of the simulation box in fm
double L = 40;

// potential to use (see params.txt for descrptions
int potential = 0;

// the particle mass (or reduced mass) in 1/fm; conversion from GeV done at runtime
double m = 1;

// couloumb coupling
double alpha = 1;

// gammmahat; modifies real part
double gam = 0;

// kappahat; generates imaginary part
double kappa = 0;

// mdfac; multiplicative factor for md
double mdfac = 1;

// initial condition type, quantum numbers n and l, and initial color state
int initType = 0;
int initN = 1;
int initL = 0;
int initC = 0;

// projection type
int projType = 0;

// time step (should be computed based on params at runtime
double dt = 0.001;

// type of derivative operator to use
int derivType = 0;

// number of threads to use for fft
int nThreads = 1;

// maximum number of time steps
int maxSteps = 10000;

// snapshot frequency
int snapFreq = 1000, snapPts= 1024;

// number of basis states to project with
int nBasis = 6;

// temperature Evolution type
int temperatureEvolution = 1;

// initial and final temperatures in GeV and proper time in GeV^{-1}.
double T0 = 0.6;
double Tf = 0.18;
double t0 = 0;
double tmed = 1; // time to turn on the medium; prior to this time T=0

// initial gaussian width
double initWidth = 0.2;

// temperature file
string temperatureFile;

// jump flag (to be turned on later)
int doJumps = 1;

// maximum initial random number
double rMax = 1;

// max jumps
int maxJumps = 999999;

// use seed in directory name?
int dirnameWithSeed = 1;

// save wavefunctions?
int saveWavefunctions = 0;

// output summary file?
int outputSummaryFile = 1;

// basis functions file
string basisFunctionsFile = "./input/basisfunctions.tsv";

// random seed
long unsigned int randomseed = 19691123;

// stepper
int stepper = 0; // default is to use LO Suzuki Trotter

// ouputfile unique id
string outfileID;

// ============================================================================
// Global vars that are loaded at runtime but accessible to outside
// ============================================================================

// these are computed in paramreader
double dx;
double dk;

// basis functions to initialize and/or project with
dcomp *basisFunctions;

// kernels
dcomp *spaceKernel,*momKernel;

// wavefunction storage for initial, tmed, and evolving
dcomp *wfnc;
dcomp *wfncInit;
dcomp *wfncSave;

// meta data associated with this trajectory
vector<string> metadata;

// temperaure along this trajectory
double *T,*ax,*ay,*az,*Lam;

// initial and evolving singlet overlaps
double *singletOverlaps0;
double *singletOverlaps;
double *singletOverlapsNoJumps;

// nmax determined from nbasis
int nmax;

// time index when medium evolution begins; determined at run time
int nmed=0;

// output file for summary
ofstream outputFile;

// store the color and angular momentum states
int cstate = 0;
int lval = 0;

// stores first random number generated for jump determination
double firstRandomNumber = 0;

// keeps track of whether evolution is lo or nlo
int NLO = 1;

// ============================================================================
// Global vars that are loaded at runtime but only accessible here
// ============================================================================

int mem_size_dcomp;
double *in,*out;
fftw_plan p;
vector<SnapShot> snaps; // Snapshot class defined in header file

// ============================================================================
//  Functions
// ============================================================================

// initialize backgrond hydro fields
// returns the time index for the time when T < Tf
int initializeTrajectoryFields(double *T, vector<string> *metadata) {
    double t = t0;
    double myT = T0; // default for Bjorken type evolution
    double myax;
    double myay;
    double myaz;
    double myLam;
    int j = 0;
    if (temperatureEvolution==1 || temperatureEvolution==2) {
        loadInterpolation(temperatureFile,metadata);
        myT = interpolateT(tmed*HBARC);
        myax = interpolateax(tmed*HBARC);
        myay = interpolateay(tmed*HBARC);
        myaz = interpolateaz(tmed*HBARC);
        myLam = interpolateLam(tmed*HBARC);
    }
    if (temperatureEvolution==0 || temperatureEvolution==3) {
        while ((t<=tmed || myT > Tf) && j<maxSteps) {
            t = t0 + j*dt;
            if (temperatureEvolution==0) myT = T[j] = (t>=tmed ? T0*pow(tmed/t,1./3.) : 0); // set T to zero before tm and then afterwards ideal Bjorken
            if (temperatureEvolution==3) myT = T[j] = (t>=tmed ? T0 : 0); // set T to zero before tm and then afterwards constant
            ax[j] = 1;
            ay[j] = 1;
            az[j] = 1;
            Lam[j] = myT;
            j++;
        }
    } else {
        for (int i=0; i<maxSteps; i++) {
            t = t0 + i*dt;
            myT = (t>=tmed ? interpolateT(t*HBARC) : 0);
            myT = T[i] = (myT>Tf ? myT : 0);
            ax[i] = (myT==0 ?  1 : interpolateax(t*HBARC));
            ay[i] = (myT==0 ?  1 : interpolateay(t*HBARC));
            az[i] = (myT==0 ?  1 : interpolateaz(t*HBARC));
            Lam[i] = (myT==0 ?  0 : interpolateLam(t*HBARC));
        }
        double Tval = 0;
        j = maxSteps;
        while (Tval==0 && j>0) {
            j--;
            Tval = T[j];
        }
    }
    if (j==maxSteps) {
        cout << "==> Warning: Max steps hit before the lowest temperature reached" << endl;
    }
    if (temperatureEvolution>0) freeInterpolation();
    return j;
}

// computes overlap squared of two wave functions
inline double computeOverlapProb(dcomp *wfnc1, dcomp *wfnc2) {
    dcomp o = 0;
    for (int i=0; i<num;i++) o += conj(wfnc1[i])*wfnc2[i];
    o *= dx;
    return real(o)*real(o)+imag(o)*imag(o);
}

// loads array with the singlet overlaps all states up to nmax
inline void loadNormedBasisOverlaps(dcomp *my_wfnc, double norm, double* oArray, dcomp* basis, int nmax, int ctype, int lval) {
    int cnt = 0;
    for (int n=1; n<=nmax; n++)
    for (int l=0; l<n; l++) {
        oArray[cnt] = ((ctype==SINGLET || ctype==ABELIAN) && l==lval ? computeOverlapProb(my_wfnc,&basis[cnt*num])/norm : 0 );
        cnt++;
    }
}

// used for loading individual coulomb wavefunctions
void loadCoulombWavefunction(dcomp* cwfnc, int n, int l)
{
    double r;
    // load the initial wavefunction into host wavefunction
    for (int i=0;i<num;i++) {
        r = (i+1)*dx;
        r *= 2*m*alpha/n;
        cwfnc[i] = dcomp(pow(r,l+1)*exp(-0.5*r)*gsl_sf_laguerre_n(n-l-1, 2*l+1, r),0);
    }
    // normalize wavefunction
    normalizeWavefunction(cwfnc);
}

// loads singlet Coulomb basis states
void loadCoulombBasisStates(dcomp* basis, int nMax) {
    int cnt = 0;
    for (int n=1; n<=nMax; n++) {
        for (int l=0; l<n; l++) {
            loadCoulombWavefunction(&basis[cnt*num],n,l);
            cnt++;
        }
    }
}

void loadBasisStatesFromFile(dcomp* basis) {
    cout << "==> Loading basis functions from file" << endl;
    // read the data
    ifstream inputFile;
    string line;
    inputFile.open(basisFunctionsFile);
    int i = 0;
    while (getline(inputFile, line))
    {
        istringstream ss(line);
        double rp,ip;
        ss.precision(numeric_limits<double>::digits10+2);
        ss >> rp >> ip;
        basis[i] = dcomp(rp,ip);
        i++;
        if (i > num*nBasis) {
            cout << "WARNING!  Basis functions file contains too many points." << endl;
            break;
        }
    }
    inputFile.close();
}

void initializeEvolution() {
    
    // set initial color and angular momentum
    cstate = initC;
    lval = initL;
    
    // create output directory; if it already exists, nothing is done
    int dstatus = createOutputDirectory(dirnameWithSeed, randomseed);
    if (dstatus!=1) {
        cout << "==> Error creating output directory" << endl;
    }
    
    if (outputSummaryFile==1) {
        // open summary output file
        char fname[512];
        snprintf(fname,512,"%s/summary.tsv",outputDirName);
        outputFile.open(fname, ios::out);
    }
    
    // load temperature and anisotropies
    T = (double *)malloc(sizeof(double)*maxSteps);
    ax = (double *)malloc(sizeof(double)*maxSteps);
    ay = (double *)malloc(sizeof(double)*maxSteps);
    az = (double *)malloc(sizeof(double)*maxSteps);
    Lam = (double *)malloc(sizeof(double)*maxSteps);
    int maxIdx = initializeTrajectoryFields(T,&metadata);
    maxSteps = (maxSteps>=maxIdx ? maxIdx : maxSteps);
    cout << "==> Set maxSteps to " << maxSteps << endl;
    nmed=0;
    while (T[nmed]==0 && nmed<maxSteps) nmed++; // determine index of in-medium evolution start
    
    // setup for state projections
    nmax = floor(-0.5 + sqrt(0.25 + 2 * (nBasis-1)))+1;
    basisFunctions = (dcomp *)malloc(nBasis*mem_size_dcomp);
    
#if LIGHTWEIGHT == 0
    if (initType==100 || projType==1) {
        loadBasisStates(basisFunctions,0); // calls eigensolver routine
        outputBasisFunctions();
    }
    else if (initType==200 || projType==2) loadBasisStatesFromFile(basisFunctions); // loads from file
    else loadCoulombBasisStates(basisFunctions,nmax);
#else
    if (initType==200 || projType==2) loadBasisStatesFromFile(basisFunctions); // loads from file
    else loadCoulombBasisStates(basisFunctions,nmax);
#endif
    
    cout << "==> Basis functions loaded" << endl;
    
    singletOverlaps0 = (double *)malloc(sizeof(double)*nBasis);
    singletOverlaps = (double *)malloc(sizeof(double)*nBasis);
    singletOverlapsNoJumps = (double *)malloc(sizeof(double)*nBasis);
    
    // kernel and wavefunction setup
    spaceKernel = (dcomp *)malloc(mem_size_dcomp);
    momKernel = (dcomp *)malloc(mem_size_dcomp);
    wfnc = (dcomp *)malloc(mem_size_dcomp);
    wfncSave = (dcomp *)malloc(mem_size_dcomp);
    wfncInit = (dcomp *)malloc(mem_size_dcomp);
    
    if (stepper==0) {
        loadSpaceKernel(T[0],ax[0],ay[0],az[0],Lam[0]); // not really necessary
        loadMomKernel();
    }
    loadInitialWavefunction();
    cout << "==> Initial wavefunction loaded" << endl;
    
    // calculate initial overlaps
    double norm = computeNorm(wfnc); // should be normalized, but to be safe
    loadNormedBasisOverlaps(wfnc, (doJumps==1 ? norm : 1), singletOverlaps0, basisFunctions, nmax, initC, initL);
    
    // dst setup
    in = (double*) fftw_malloc(sizeof(double)*num);
    out = (double*) fftw_malloc(sizeof(double)*num);
    fftw_init_threads();
    fftw_plan_with_nthreads(nThreads);
    p = fftw_plan_r2r_1d(num,in,out,FFTW_RODFT00,FFTW_ESTIMATE);
    
    // seed the RNG
    srand(randomseed);
    
    // setup static matrices for Crank-Nicholson (CN)
    if (stepper>0) setupCN();
    
    // print header
    print_header(nBasis);
}

void finalizeEvolution() {
    // cleanup memory and threads
    if (outputSummaryFile==1) outputFile.close();
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    free(spaceKernel);
    free(momKernel);
    free(wfnc);
    free(wfncSave);
    free(wfncInit);
    free(basisFunctions);
    free(T);
    free(ax);
    free(ay);
    free(az);
    free(Lam);
    free(singletOverlaps0);
    free(singletOverlaps);
    free(singletOverlapsNoJumps);
    fftw_cleanup_threads();
}

void outputInfo(double n, double norm, bool jump) {
    const double rexp = computeRexp(wfnc,norm);
    const double evac = computeEvac(wfnc,norm);
    loadNormedBasisOverlaps(wfnc, norm, singletOverlaps, basisFunctions, nmax, cstate, lval);
    if (saveWavefunctions && !jump) outputSnapshot(wfnc,n);
    outputSummary(n, computeNorm(wfnc), cstate, lval, rexp, evac, singletOverlaps, nBasis, jump);
}

// adds wfnc to end of vector of snapshots; only done during Heff evolution
// in order to accelerate later evolution with jumps
void recordHeffSnap(int idx, dcomp* wfnc, double norm) {
    SnapShot s;
    s.setData(wfnc);
    s.norm = norm;
    s.timeIdx = idx;
    snaps.push_back(s);
}

// Heff evolution
double evolveWavefunction(int nStart, int nSteps) {
    
    double norm = 1;
    double rexp;
    
    // setup for some operators
    if (stepper==0) setupCN();
    
    // begin time loop
    for (int n=nStart;n<nStart+nSteps;n++) {
        
        
        // update the space kernel
        if (stepper==0) loadSpaceKernel(T[n],ax[n],ay[n],az[n],Lam[n]);
        
        // output info to screen and/or disk
        if (n%snapFreq==0 || n==nStart) {
            norm = computeNorm(wfnc);
	    outputInfo(n,(doJumps==1?norm:1),false);
	    //outputInfo(n,(doJumps==1?norm:1),(doJumps==1?true:false));
#if FASTFORWARD == 1
            recordHeffSnap(n,wfnc,norm);
#endif
        }
        
        // update the wavefunction
        switch(stepper) {
            case 0: // ST
                makeStep(p,wfnc,in,out,spaceKernel,momKernel);
                break;
            case 1: // CN
                makeStepCN(wfnc,T[n]);
                break;
            case 2: // CN NLO
                makeStepCN_NLO(wfnc,T[n]);
                break;
            default:
                cout << "Unknown stepper.  Exiting" << endl;
                exit(-1);
        }
        
    } // end time loop
    
    // output final wavefunction, summary info, and ratios
    norm = computeNorm(wfnc);
    outputInfo(nStart+nSteps,(doJumps==1?norm:1),false);
    //outputInfo(nStart+nSteps,(doJumps==1?norm:1),(doJumps==1?true:false));
    return norm;
}

// used for testing
int listState=0;
double list[200] = {0.902004, 0.0771218, 0.526584, 0.716861, 0.850578, 0.816051, \
    0.66174, 0.628967, 0.72624, 0.846015, 0.833946, 0.748876, 0.61051, \
    0.306054, 0.906464, 0.67328, 0.169597, 0.193307, 0.356879, 0.992418, \
    0.84201, 0.339602, 0.75379, 0.627999, 0.368233, 0.242294, 0.63597, \
    0.67781, 0.324625, 0.675205, 0.83352, 0.517339, 0.675978, 0.335518, \
    0.444907, 0.547719, 0.520426, 0.239918, 0.332194, 0.119913, 0.492081, \
    0.175313, 0.577837, 0.493531, 0.854449, 0.570509, 0.0395022, \
    0.623516, 0.536077, 0.0446373, 0.985748, 0.840046, 0.296121, \
    0.446471, 0.846025, 0.724239, 0.259125, 0.192353, 0.664467, 0.937283, \
    0.330466, 0.967336, 0.834368, 0.540503, 0.179253, 0.760045, 0.744041, \
    0.0859915, 0.886738, 0.719458, 0.37563, 0.469095, 0.357855, 0.273017, \
    0.524416, 0.483008, 0.444057, 0.394602, 0.400391, 0.978413, 0.829558, \
    0.0974571, 0.8213, 0.691098, 0.564818, 0.93643, 0.859949, 0.208832, \
    0.424529, 0.41145, 0.0422367, 0.896364, 0.181662, 0.228145, 0.83876, \
    0.240425, 0.259162, 0.673233, 0.536114, 0.731895, 0.367546, 0.670294, \
    0.692297, 0.966317, 0.92969, 0.307716, 0.998545, 0.0141067, \
    0.00140062, 0.473301, 0.990136, 0.835935, 0.023593, 0.773398, \
    0.961369, 0.801838, 0.401233, 0.325588, 0.789543, 0.965013, 0.105901, \
    0.914193, 0.815502, 0.51082, 0.366306, 0.74702, 0.325765, 0.417312, \
    0.531775, 0.192823, 0.442802, 0.844184, 0.558917, 0.58526, 0.36905, \
    0.0675507, 0.910732, 0.160915, 0.972648, 0.534424, 0.595779, \
    0.0488006, 0.304066, 0.0921461, 0.932946, 0.447758, 0.355093, \
    0.45659, 0.391351, 0.895639, 0.411195, 0.57124, 0.0486459, 0.992874, \
    0.105776, 0.984092, 0.749592, 0.0731683, 0.59347, 0.829591, 0.475579, \
    0.953434, 0.395708, 0.461299, 0.366201, 0.898504, 0.541738, \
    0.0944358, 0.10248, 0.68872, 0.144209, 0.00524146, 0.0536605, \
    0.859969, 0.682222, 0.153432, 0.802234, 0.354705, 0.737036, 0.833627, \
    0.695842, 0.466092, 0.528022, 0.92577, 0.714757, 0.455498, 0.935318, \
    0.448192, 0.0181908, 0.298858, 0.23926, 0.190831, 0.473549, 0.177332, \
    0.888072, 0.273336, 0.127446, 0.292276, 0.936197, 0.576544};

inline double listRandom() {
    double val = list[listState];
    listState = (listState+1)%200;
    cout << "random = " << val << endl;
    return val;
}

// generates a random number between 0 and 1
inline double Random() {
    if (randomseed==-1) return listRandom();
    double val = rand()/((double) RAND_MAX);
    //cout << "Random = " << val << endl;
    return val;
}

// multiplies by the jump operator and normalizes the resulting wavefunction
// this is LO jump operator
inline void doJump(dcomp* wfnc)
{
    for (int i=0;i<num;i++) wfnc[i] *= (double)(i+1);
    normalizeWavefunction(wfnc);
}

// computes probability for jump from one angular momentum state to another
double AngMomProb(int l) {
    if (l==0) return 0.;
    else return ((2.*l-1.)/(2.*l+1.))*(1.-AngMomProb(l-1));
}

void saveNoJumpsSingletOverlaps() {
    for (int i=0; i<nBasis; i++) {
        singletOverlapsNoJumps[i] = singletOverlaps[i];
    }
}

void retrieveNoJumpsSingletOverlaps() {
    for (int i=0; i<nBasis; i++) singletOverlaps[i] = singletOverlapsNoJumps[i];
}

void fastForwardHeffEvolution(double r, double *norm, int *nStart){
    // find place to jump to in snapshots
    double tNorm = 1;
    int tSnap = 0;
    SnapShot s;
    while (tNorm > r && tSnap < snaps.size()) {
        s = snaps[tSnap];
        tNorm = s.norm;
        tSnap++;
    }
    s = snaps[tSnap-2];
    *nStart = s.timeIdx;
    *norm = s.norm;
    s.getData(wfnc);
}

void evolveWavefunctionWithJumps(int nStart, double rmin) {
    
    int nJumps = 0;
    double norm = 1;
    double rexp;
    
    // setup for some operators
    setupCN();
    
    // generate initial random number
    double r = Random();
    firstRandomNumber = r; // save this for testing output

#if OPT_EVOL == 1
    // check to see if jumps will be triggered at all
    if (r<rmin) {
        print_line();
        cout << "==> No jumps necessary.  Setting overlaps to Heff overlaps." << endl;
        retrieveNoJumpsSingletOverlaps();
        return;
    }
    // check to see if upper threshold is crossed
    if (r > rMax) {
        print_line();
        cout << "==> Initial random number greater than rmax.  Setting overlaps to zero." << endl;
        for (int i=0; i<nBasis; i++) singletOverlaps[i] = 0;
        return;
    }
    
#if FASTFORWARD == 1
    // fast forward
    fastForwardHeffEvolution(r,&norm,&nStart);
#endif
#endif
    
    // begin time loop
    int n=nStart;
    while (n<maxSteps) {
        
        // normal evolution
        while (n<maxSteps && r < norm) {
            
            // load the kernel
            loadSpaceKernel(T[n],ax[n],ay[n],az[n],Lam[n]);
            
            // output snapshot to disk every snapFreq steps; also compute wavefunction norm and output that to screen
            if (n%snapFreq==0 || n==nStart) outputInfo(n,norm,false);
            
            makeStep(p,wfnc,in,out,spaceKernel,momKernel);
            
            norm = computeNorm(wfnc);
            
            // increment the time counter
            n++;
        }
        
        // do the jump if necessary
        if (n<maxSteps) {
            
            // output some info just before the jump
            norm = computeNorm(wfnc);
            outputInfo(n,norm,true);
            
            // determine new angular momentum state
            if (Random() < AngMomProb(lval)) lval -= 1;
            else lval += 1;
            
            // determine new color state
            if (cstate == OCTET  && Random() < 2./7.) cstate = SINGLET;
            else cstate = OCTET;
            
            // apply jump operator and normalize to 1
            doJump(wfnc);
            
            // generate new random # and increment nJumps
            r = Random();
            nJumps++;
            
            // output some info just after the jump
            norm = 1; // forced to be one by doJump so don't waste time computing it
            outputInfo(n,norm,true);
        }
        
        // check to see if we exceeded the maximum number of jumps
        if (nJumps>=maxJumps) {
            print_line();
            cout << "==> Exceeded max jumps.  Terminating evolution." << endl;
            print_line();
            break;
        }
        
    } // end time loop
    
    // output final wavefunction and summary info
    norm = computeNorm(wfnc);
    outputInfo(n,norm,false);
    
}

void evolveWavefunctionWithJumpsCN(int nStart, double rmin) {
    
    int nJumps = 0;
    double norm = 1;
    double rexp;
    
    // generate initial random number
    double r = Random();
    firstRandomNumber = r; // save this for testing output

#if OPT_EVOL == 1
    // check to see if jumps will be triggered at all
    if (r<rmin) {
        print_line();
        cout << "==> No jumps necessary.  Setting overlaps to Heff overlaps." << endl;
        retrieveNoJumpsSingletOverlaps();
        return;
    }
    // check to see if upper threshold is crossed
    if (r > rMax) {
        print_line();
        cout << "==> Initial random number greater than rmax.  Setting overlaps to zero." << endl;
        for (int i=0; i<nBasis; i++) singletOverlaps[i] = 0;
        return;
    }
#if FASTFORWARD == 1
    // fast forward
    fastForwardHeffEvolution(r,&norm,&nStart);
#endif
#endif

    // turn on/off NLO terms based on stepper setting   
    NLO = stepper==2 ? 1 : 0;
 
    // begin time loop
    int n=nStart;
    while (n<maxSteps) {
        
        // normal evolution
        while (n<maxSteps && r < norm) {
            
            // output snapshot to disk every snapFreq steps; also compute wavefunction norm and output that to screen
            if (n%snapFreq==0 || n==nStart) outputInfo(n,norm,false);
            
            // update the wavefunction
            switch(stepper) {
                case 1: // CN
                    makeStepCN(wfnc,T[n]);
                    break;
                case 2: // CN NLO
                    makeStepCN_NLO(wfnc,T[n]);
                    break;
                default:
                    cout << "Unsupported stepper.  Exiting" << endl;
                    exit(-1);
            }
            
            norm = computeNorm(wfnc);
            
            // increment the time counter
            n++;
        }
        
        // do the jump if necessary
        if (n<maxSteps) {
            
            double r1 = Random();
            double r2 = Random();
            int jt = 0;
            
            // output some info just before the jump
            norm = computeNorm(wfnc);
            outputInfo(n,norm,true);
            
            if (cstate==SINGLET) {
                cstate = 1;
                if (r1 < AngMomProb(lval)) {
                    lval -= 1;
                    jt = 1;
                }
                else {
                    lval += 1;
                    jt = 2;
                }
            } else if (cstate==ABELIAN) {
                if (r1 < AngMomProb(lval)) {
                    lval -= 1;
                    jt = 5;
                }
                else {
                    lval += 1;
                    jt = 6;
                }
            } else {
                
                double P1 = GammaOOdown(wfnc,T[n]);
                double P2 = GammaOOup(wfnc,T[n]);
                double P3 = GammaOSdown(wfnc,T[n]);
                double P4 = GammaOSup(wfnc,T[n]);
                double Psum = P1+P2+P3+P4;
                
                P1 /= Psum; P2 /= Psum; P3 /= Psum; P4 /= Psum;
                
                if (r1<P1+P3) {
                    lval -= 1;
                    jt = -1;
                } else {
                    lval += 1;
                    jt = 1;
                }
                if (r2<P3+P4) {
                    cstate = 0;
                    jt = (jt==-1 ? 3 : 4);
                } else {
                    cstate  = 1;
                    jt = (jt==-1 ? 5 : 6);
                }
            }
            
            // apply jump operator and normalize to 1
            doJump(wfnc,T[n],jt);
            
            // output some info just after the jump
            norm = 1; // forced to be one by doJump so don't waste time computing it
            outputInfo(n,norm,true);
            
            // generate new random # and increment nJumps
            r = Random();
            nJumps++;
        }
        
        // check to see if we exceeded the maximum number of jumps
        if (nJumps>=maxJumps) {
            print_line();
            cout << "==> Exceeded max jumps.  Terminating evolution." << endl;
            print_line();
            break;
        }
        
    } // end time loop
    
    // output final wavefunction and summary info
    norm = computeNorm(wfnc);
    outputInfo(n,norm,false);
    
}
