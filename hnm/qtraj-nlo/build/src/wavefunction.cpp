/*
 
 wavefunction.cpp
 
 Copyright (c) Michael Strickland and Anurag Tiwari
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdio.h>

#define ARMA_USE_SUPERLU 1
#include <armadillo>

using namespace arma;
using namespace std;

#include "potential.h"
#include "initialcondition.h"
#include "trajectory.h"
#include "wavefunction.h"

double MQ;

sp_cx_mat Id;
sp_cx_mat K1;
sp_cx_mat K2;
sp_cx_mat X;
sp_cx_mat Xin;
sp_cx_mat X2;
sp_cx_mat Xin2;
sp_cx_mat P;
sp_cx_mat XPC;
sp_cx_mat Heff;
sp_cx_mat HeffRe;
sp_cx_mat Gam;
sp_cx_mat Potential;
sp_cx_mat ML;
sp_cx_mat MR;

// Computes Exp(-I x dt)
inline dcomp kernelFunc(dcomp x) {
    // multiply by -I dt and exponentiate
    return exp(dcomp(imag(x)*dt,-real(x)*dt));
}

// compute norm
double computeNorm(dcomp* wfnc)
{
    double N = 0;
    for (int i=0;i<num;i++) N += real(conj(wfnc[i])*wfnc[i]);
    return N*dx;
}

void normalizeWavefunction(dcomp* wfnc) {
    double N = computeNorm(wfnc);
    N = 1/sqrt(N);
    for (int i=0;i<num;i++) wfnc[i] *= N;
}

// compute <r>/s0
double computeRexp(dcomp* wfnc, double norm)
{
    double rexp = 0;
    for (int i=0;i<num;i++) rexp += real(conj(wfnc[i])*wfnc[i])*(i+1);
    return rexp*dx*dx*(alpha*m*2./3.)/norm; // scale by 1s <r>
}

// compute <psi|Hvac|psi>/<psi|psi>
double computeEvac(dcomp* wfnc, double norm)
{
    const cx_vec psi(wfnc,num);
    sp_cx_mat Hvac(num,num);
    const double angFac = 0.5*lval*(lval+1)/m; // reduced mass here
    if (cstate==0) { // singlet
        Hvac = K1 - alpha*Xin + angFac*Xin2;
    } else { // octet
        Hvac = K1 + (alpha/(NC*NC-1))*Xin + angFac*Xin2;
    }
    return real(dot(conj(psi),Hvac*psi))/norm;
}

// setup the space kernel
void loadSpaceKernel(double T, double ax, double ay, double az, double Lam)
{
    for (int i=0;i<num;i++) spaceKernel[i] = kernelFunc(0.5*Veff((i+1)*dx,T,ax,ay,az,Lam));
}

// setup the mom kernel and load to device
void loadMomKernel()
{
    for (int i=0;i<num;i++)  {
        const double k = (i+1)*dk;
        if (derivType==0) momKernel[i] = kernelFunc(dcomp(0.5*k*k/m,0));
        else momKernel[i] = kernelFunc(dcomp((1-cos(k*dx))/m/dx/dx,0));
    }
}

// setup intial wave function
void loadInitialWavefunction()
{
    int initState = 0.5*initN*(initN-1)+initL; // this comes from flattening the triangle
    if (initType==100 && initState>nBasis-1) {
        cout << "initState not in basis elements loaded.  Increase nBasis." << endl;
        exit(-1);
    }
    for (int i=0;i<num;i++) {
        wfnc[i] = basisFunctions[initState*num+i];
        if (initType<100)
            wfnc[i] = initialWavefunction((i+1)*dx, initN, initL);
        else
            wfnc[i] = basisFunctions[initState*num+i];
    }
    // normalize wavefunction
    normalizeWavefunction(wfnc);
    // save for convenient access later
    for (int i=0;i<num;i++) wfncInit[i] = wfnc[i];
}

// Discrete Sine Transform
inline void DST(fftw_plan p, dcomp* wfnc, double* in, double* out)
{
    // load real part
    for (int i=0;i<num;i++) in[i] = real(wfnc[i]);
    // DST
    fftw_execute(p);
    // save real part in real part of wfnc
    for (int i=0;i<num;i++) wfnc[i] = dcomp(out[i]/sqrt(2*(num + 1)),imag(wfnc[i]));
    // load imaginary part
    for (int i=0;i<num;i++) in[i] = imag(wfnc[i]);
    // DST
    fftw_execute(p);
    // save imag part in the imag part of wfnc
    for (int i=0;i<num;i++) wfnc[i] = dcomp(real(wfnc[i]),out[i]/sqrt(2*(num + 1)));
    return;
}

// LO Suzuki-Trotter Step
void makeStep(fftw_plan p, dcomp* wfnc, double* in, double* out, dcomp* spaceKernel, dcomp* momKernel)
{
    // make one spatial half-step
    for (int i=0; i<num; i++) wfnc[i] = spaceKernel[i]*wfnc[i];
    // forward DST transform
    DST(p, wfnc, in, out);
    // make one full step in momentum space
    for (int i=0; i<num; i++) wfnc[i] = momKernel[i]*wfnc[i];
    // backward DST transform; it's its own inverse
    DST(p, wfnc, in, out);
    // make one spatial half-step
    for (int i=0; i<num; i++) wfnc[i] = spaceKernel[i]*wfnc[i];
}

// Setup for Crank-Nicholson Steppers
void setupCN() {
    
    // static
    Id.resize(num,num);   // identity matrix
    K1.resize(num,num);   // p_r^2/2/m (m = reduced mass)
    K2.resize(num,num);   // p_r^2
    X.resize(num,num);    // r
    Xin.resize(num,num);  // 1/r
    X2.resize(num,num);   // r^2
    Xin2.resize(num,num); // 1/r^2
    P.resize(num,num);    // p_r
    
    double lf = 1/(dx*dx*m);
    for (int i=0;i<num;i++) {
        // Id
        Id(i,i) = 1;
        // K1
        K1(i,i) = lf; //  1/m/dx/dx
        if (i>0) K1(i,i-1) = -lf/2; // - 1/2/m/dx/dx
        if (i<num-1) K1(i,i+1) = -lf/2; // - 1/2/m/dx/dx
        // X, Xin
        double x = (i+1)*dx;
        X(i,i) = x;
        Xin(i,i) = 1/x;
        // X^2, Xin^2
        X2(i,i) = x*x;
        Xin2(i,i) = 1/x/x;
        // P operator
        if (i>0) P(i,i-1) = I/(2*dx);
        if (i<num-1) P(i,i+1) = -I/(2*dx);
    }
    
    // derived from above
    K2 = 2*m*K1;
    XPC = X*P + P*X;

    // quark mass in terms of reduced mass
    MQ = 2*m;
    
    // dynamic
    Heff.resize(num,num);
    HeffRe.resize(num,num);
    Gam.resize(num,num);
    Potential.resize(num,num);
    ML.resize(num,num);
    MR.resize(num,num);
}

// LO Crank-Nicholson Step
void makeStepCN(dcomp* wfnc, double T) {
    
    const cx_vec psi0(wfnc,num);
    cx_vec psi1(num);
    
    for (int i=0;i<num;i++) Potential(i,i) = Veff((i+1)*dx,T);
    Heff = K1 + Potential;
    
    // calculate ML and MR
    ML = Id + 0.5*I*Heff*dt;
    MR = Id - 0.5*I*Heff*dt;
    
    psi1 = spsolve(ML, MR*psi0);
    
    for (int i=0;i<num;i++) wfnc[i] = psi1(i);
    
}

// NLO Crank-Nicholson Step
void makeStepCN_NLO(dcomp* wfnc, double T) {
    
    const cx_vec psi0(wfnc,num);
    cx_vec psi1(num);
    
    const double T3 = T*T*T;
    const double T2 = T*T;
    const double angFac = 0.5*lval*(lval+1)/m; // reduced mass here
    const double xFac = 0.25*NC*alpha*T2/CF;
    const double idFac = -1.5*T2/MQ;
    const double vFac = 0.125*NC*T*alpha/MQ/CF;
    const double idFac2 = T*pow(0.125*NC*alpha/CF,2);
    const double K2Fac = 0.25*T/MQ/MQ;
    const double xpcFac = 0.25*kappaT(T)*T2/MQ;
    
    if (cstate==0) { // singlet
        HeffRe = K1 - alpha*Xin + angFac*Xin2 + 0.5*gam*T3*X2 + NLO*xpcFac*XPC;
        Gam = kappaT(T)*(T3*X2 + NLO*(idFac*Id - xFac*X + K2Fac*(K2 + lval*(lval+1)*Xin2) + vFac*Xin + idFac2*Id));
    } else if (cstate==1) { // octet
        HeffRe = K1 + (alpha/(NC*NC-1))*Xin + angFac*Xin2 + (0.5*(NC*NC-2)/(NC*NC-1))*(0.5*gam*T3*X2 + NLO*xpcFac*XPC);
        Gam = (kappaT(T)/(NC*NC-1))*(T3*X2 + NLO*(idFac*Id + xFac*X + K2Fac*(K2 + lval*(lval+1)*Xin2) - vFac*Xin + idFac2*Id)) + (0.5*kappaT(T)*(NC*NC-4)/(NC*NC-1))*(T3*X2 + NLO*(idFac*Id + K2Fac*(K2 + lval*(lval+1)*Xin2)));
    } else {
        HeffRe = K1 - (alpha)*Xin + angFac*Xin*Xin + (0.5*gam*T3*X2 + NLO*xpcFac*XPC);
        Gam = kappaT(T)*(T3*X2 + NLO*(idFac*Id + K2Fac*(K2 + lval*(lval+1)*Xin2)));
    }
    
    Heff = HeffRe - 0.5*I*Gam;
    
    // calculate ML and MR
    ML = Id + 0.5*I*Heff*dt;
    MR = Id - 0.5*I*Heff*dt;
    
    psi1 = spsolve(ML, MR*psi0);
    
    for (int i=0;i<num;i++) wfnc[i] = psi1(i);
}

// P1
double GammaOOdown(dcomp *wfnc, double T) {
    double T3 = T*T*T;
    double T2 = T*T;
    cx_vec psi(wfnc,num);
    sp_cx_mat GamOOdown(num,num);
    GamOOdown = kappaT(T)*(0.5*(NC*NC-4)/(NC*NC-1))*(lval/(2*lval+1.))*(T3*X2 + NLO*(-T2*(3/(2*MQ))*Id + (T/4/MQ/MQ)*(K2 + lval*(lval+1)*Xin2)));
    return real(dot(conj(psi),GamOOdown*psi));
}

// P2
double GammaOOup(dcomp *wfnc, double T) {
    double T3 = T*T*T;
    double T2 = T*T;
    cx_vec psi(wfnc,num);
    sp_cx_mat GamOOup(num,num);
    GamOOup = kappaT(T)*(0.5*(NC*NC-4)/(NC*NC-1))*((lval+1.)/(2*lval+1.))*(T3*X2 + NLO*(-T2*(3/(2*MQ))*Id + (T/4/MQ/MQ)*(K2 + lval*(lval+1)*Xin2)));
    return real(dot(conj(psi),GamOOup*psi));
}

// P3
double GammaOSdown(dcomp *wfnc, double T) {
    double T3 = T*T*T;
    double T2 = T*T;
    cx_vec psi(wfnc,num);
    sp_cx_mat GamOSdown(num,num);
    GamOSdown = kappaT(T)*(1/(NC*NC-1))*(lval/(2*lval+1.))*(T3*X2 + NLO*(-T2*(3/(2*MQ))*Id + (NC*alpha*T2/(4*CF))*X + (T/(4*MQ*MQ))*(K2 + lval*(lval+1)*Xin2) + (-1*NC*alpha*T/(8*MQ*CF))*Xin + T*pow((NC*alpha/(8*CF)),2)*Id));
    return real(dot(conj(psi),GamOSdown*psi));
}

// P4
double GammaOSup(dcomp *wfnc, double T) {
    double T3 = T*T*T;
    double T2 = T*T;
    cx_vec psi(wfnc,num);
    sp_cx_mat GamOSup(num,num);
    GamOSup = kappaT(T)*(1/(NC*NC-1))*((lval+1.)/(2*lval+1.))*(T3*X2 + NLO*(-T2*(3/(2*MQ))*Id + (NC*alpha*T2/(4*CF))*X + (T/(4*MQ*MQ))*(K2 + lval*(lval+1)*Xin2) + (-1*NC*alpha*T/(8*m*CF))*Xin + T*pow((NC*alpha/(8*CF)),2)*Id));
    return real(dot(conj(psi),GamOSup*psi));
}

//
// Note: All jump operators are multiplied by T; normalization is arbitrary
//

// 1 = downSO
inline sp_cx_mat jumpDownSO(double T) {
    return T*X + NLO*((-NC*alpha/8/CF)*Id + (0.5/MQ)*(I*P + lval*Xin));
}

// 2 = upSO
inline sp_cx_mat jumpUpSO(double T) {
    return T*X + NLO*((-NC*alpha/8/CF)*Id + (0.5/MQ)*(I*P - (lval+1)*Xin));
}

// 3 = downOS
inline sp_cx_mat jumpDownOS(double T) {
    return T*X + NLO*((NC*alpha/8/CF)*Id + (0.5/MQ)*(I*P + lval*Xin));
}

// 4 = upOS
inline sp_cx_mat jumpUpOS(double T) {
    return T*X + NLO*((NC*alpha/8/CF)*Id + (0.5/MQ)*(I*P - (lval+1)*Xin));
}

// 5 = downOO
inline sp_cx_mat jumpDownOO(double T) {
    return T*X + NLO*(0.5/MQ)*(I*P + lval*Xin);
}

// 6 = upOO
inline sp_cx_mat jumpUpOO(double T) {
    return T*X + NLO*(0.5/MQ)*(I*P - (lval+1)*Xin);
}

// multiplies by the jump operator and normalizes the resulting wavefunction
void doJump(dcomp* wfnc, double T, int jumpType)
{
    sp_cx_mat C(num,num);
    const cx_vec psi(wfnc,num);
    cx_vec psi1(num);
    switch (jumpType) {
        case 1:
            //cout << "Jump type: downSO" << endl;
            C = jumpDownSO(T); break;
        case 2:
            //cout << "Jump type: upSO" << endl;
            C = jumpUpSO(T); break;
        case 3:
            //cout << "Jump type: downOS" << endl;
            C = jumpDownOS(T); break;
        case 4:
            //cout << "Jump type: upOS" << endl;
            C = jumpUpOS(T); break;
        case 5:
            //cout << "Jump type: downOO" << endl;
            C = jumpDownOO(T); break;
        case 6:
            //cout << "Jump type: upOO" << endl;
            C = jumpUpOO(T); break;
        default:
            cout << "Unknown jump type. Exiting." << endl;
            exit(-1); break;
    }
    psi1 = C*psi;
    for (int i=0;i<num;i++) wfnc[i] = psi1(i);
    normalizeWavefunction(wfnc);
}
