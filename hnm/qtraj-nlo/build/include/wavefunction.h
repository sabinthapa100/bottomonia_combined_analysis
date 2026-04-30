/*
 
 wavefunction.h
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
*/

#ifndef __wavefunction_h__
#define __wavefunction_h__

#include <complex>
#include <fftw3.h>

typedef std::complex<double> dcomp;

const dcomp I(0,1);

double computeNorm(dcomp* wfnc);

void normalizeWavefunction(dcomp* wfnc);
double computeRexp(dcomp* wfnc, double norm);
double computeEvac(dcomp* wfnc, double norm);

void loadSpaceKernel(double T, double ax, double ay, double az, double Lam);
void loadMomKernel();
void loadInitialWavefunction();

void makeStep(fftw_plan p, dcomp* wfnc, double* in, double* out, dcomp* spaceKernel, dcomp* momKernel);
void makeStepCN(dcomp* wfnc, double T);
void makeStepCN_NLO(dcomp* wfnc, double T);

void setupCN();

double GammaOOdown(dcomp *wfnc, double T);
double GammaOOup(dcomp *wfnc, double T);
double GammaOSdown(dcomp *wfnc, double T);
double GammaOSup(dcomp *wfnc, double T);

void doJump(dcomp* wfnc, double T, int jumpType);

#endif /* __wavefunction_h__ */
