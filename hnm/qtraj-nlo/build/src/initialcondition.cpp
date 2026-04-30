/*
 
 initialcondition.cpp
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <gsl/gsl_sf_laguerre.h>

#include "trajectory.h"

using namespace std;

long long int factorial(int n)
{
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

dcomp coloumbBasisFunction(int n, int l, double r) {
    double beta = 2./n;
    double norm = sqrt(pow(beta,3.)*factorial(n-l-1)/(2*n*factorial(n+l)));
    double rho = beta*m*alpha*r;
    return dcomp(r*norm*pow(rho,l)*exp(-0.5*rho)*gsl_sf_laguerre_n(n-l-1, 2*l+1, rho),0);
}

//
// set initial wave function
// i = grid point index
// n = principle quantum number
// l = angular momentum quantum number
//
dcomp initialWavefunction(double r, int n, int l) {
    switch(initType) {
        case 0:
            // Singlet Coulomb
            return coloumbBasisFunction(n,l,r);
            break;
        case 1:
            // gaussian delta
            r *= m*alpha; // here we divide r by a0 = 1 / alpha mu
            return dcomp(pow(r,l+1)*exp(-r*r/initWidth/initWidth),0);
            break;
        default:
            cout << "Unknown initial condition.  Exiting." << endl;
            exit(-1);
            break;
    }
}
