/*
 
 eigensolver.cpp
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "math.h"

#include <armadillo>
using namespace arma::newarp;

#include "eigensolver.h"
#include "outputroutines.h"
#include "potential.h"

using namespace std;

void computeStatesARMA(double T, double l, double *evals, double *evecs) {
    
    arma::sp_mat M(num,num);
    
    const double mass = 2*m; // mass from reduced mass
    double lapfac = 1/mass/dx/dx;
    
    for (int i=0;i<num;i++) {
        double r = (i+1)*dx;
        M(i,i) = 2*lapfac + 2*mass + real(V(r,T)) + l*(l+1)/mass/r/r;
        if (i>0) M(i-1,i) = -lapfac;
        if (i<num-1) M(i+1,i) = -lapfac;
    }

    // Construct matrix operation object
    SparseGenMatProd <double> op(M);
    
    // Construct eigen solver object, requesting nBasis eigenvalues/vectors
    SymEigsSolver< double, EigsSelect::SMALLEST_ALGE, SparseGenMatProd<double> > eigs(op, nBasis, nBasis*nBasis*nBasis);
    
    // Initialize and compute eigensystem
    eigs.init();
    
    // Retrieve results
    int nconv = eigs.compute();
    if(nconv > 0) {
        arma::vec evalues = eigs.eigenvalues();
        arma::mat evectors = eigs.eigenvectors();
        for (int i=0;i<nBasis;i++) {
            evals[i] = evalues(i);
            for (int j=0;j<num;j++)
                evecs[j+i*num] = evectors(j,i);
        }
    }
    else {
        cout << "Error: Eigensystem determination did not converge." << endl;
        exit(-1);
    }
    
}

void extractEigenvectorARMA(double* eigenvecs, dcomp* mycol, int col) {
    for (int i=0; i<num; i++)
        mycol[i] = dcomp(eigenvecs[i+col*num]/sqrt(dx),0); // load taking into account different normalization
}

void loadBasisStates(dcomp *basisFunctions, double T) {

    cout << "==> Finding the first " << nBasis << " eigenstates (Armadillo)" << endl;
    
    // l = 0, 1, 2
    double *vals0 = (double *)malloc(sizeof(double)*nBasis); // eigenvalues
    double *vecs0 = (double *)malloc(sizeof(double)*num*nBasis); // eigenvectors
    double *vals1 = (double *)malloc(sizeof(double)*nBasis); // eigenvalues
    double *vecs1 = (double *)malloc(sizeof(double)*num*nBasis); // eigenvectors
    double *vals2 = (double *)malloc(sizeof(double)*nBasis); // eigenvalues
    double *vecs2 = (double *)malloc(sizeof(double)*num*nBasis); // eigenvectors
    
    computeStatesARMA(T,0,vals0,vecs0);
    computeStatesARMA(T,1,vals1,vecs1);
    computeStatesARMA(T,2,vals2,vecs2);
    
    print_line();
    cout << "==> M(1s) = " << vals0[0] << endl;
    cout << "==> M(2s) = " << vals0[1] << endl;
    cout << "==> M(2p) = " << vals1[0] << endl;
    cout << "==> M(3s) = " << vals0[2] << endl;
    cout << "==> M(3p) = " << vals1[1] << endl;
    cout << "==> M(3d) = " << vals2[0] << endl;
    print_line();
    
    extractEigenvectorARMA(vecs0, &basisFunctions[0*num], 0);
    extractEigenvectorARMA(vecs0, &basisFunctions[1*num], 1);
    extractEigenvectorARMA(vecs1, &basisFunctions[2*num], 0);
    extractEigenvectorARMA(vecs0, &basisFunctions[3*num], 2);
    extractEigenvectorARMA(vecs1, &basisFunctions[4*num], 1);
    extractEigenvectorARMA(vecs2, &basisFunctions[5*num], 0);
    
    free(vals0);
    free(vecs0);
    free(vals1);
    free(vecs1);
    free(vals2);
    free(vecs2);
}
