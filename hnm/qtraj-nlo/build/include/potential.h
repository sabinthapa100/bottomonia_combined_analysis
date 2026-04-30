/*
 
 potential.h
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
*/

#include "trajectory.h"

#ifndef __potential_h__
#define __potential_h__

#define NC	3. // number of colors
#define NF	2. // number of flavors
#define CF 	((NC*NC-1.)/2./NC)
#define USE_RUNNING_COUPLING	1

// qcd beta function coeffs
#define B0	(11*NC-2*NF)/(12*M_PI)
#define B1	(17*NC*NC-NF*(10*NC+6*(NC*NC-1)/(2*NC))/2)/(24*M_PI*M_PI)
#define B2	(2857 - 5033*NF/9 + 325*NF*NF/27)/(128*M_PI*M_PI*M_PI)

// Lambda_MSbar in GeV
const double LAMBDA_MS=0.344;

// string breaking distance in inverse GeV
const double SIGMA=0.210; // GeV^2
const double SBDISTIGEV=1.25/HBARC; // GeV

dcomp V(double r, double T);
dcomp V(double r, double T, double ax, double ay, double az, double Lam);
dcomp Veff(double r, double T); // includes L^2 term
dcomp Veff(double r, double T, double ax, double ay, double az, double Lam); // includes L^2 term

// temperature dependent kappa function
double kappaT(double T);

#endif /* __potential_h__ */
