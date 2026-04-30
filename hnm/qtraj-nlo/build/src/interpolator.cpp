/*
 
 interpolator.cpp
 
 Copyright (c) Michael Strickland 
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <memory>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "trajectory.h"
#include "interpolator.h"

using namespace std;

int nPts;
double *fnct,*fncT, *fncax, *fncay, *fncaz, *fncLam;

double lxmin,lxmax;

gsl_interp_accel  *acc_T, *acc_ax, *acc_ay, *acc_az, *acc_Lam; 
gsl_spline *spline_T, *spline_ax, *spline_ay, *spline_az, *spline_Lam;

// read data file
void readData(string filename, vector<string> *metadata) {
    
    // read the data
    ifstream inputFile;
    string line;
    inputFile.open(filename);
    // process headers; depends on type of file
    if (temperatureEvolution==1) {
        getline(inputFile, line); // read the one-line header and discard since we don't need it
    }
    else if (temperatureEvolution==2) {
        getline(inputFile, line); // line 1; nMeta
        istringstream ss(line);
        int nMeta;
        ss >> nMeta;
        for (int i=0;i<nMeta;i++) {
            getline(inputFile, line); // metadata lines
            metadata->push_back(line);
        }
    }
    else { cout << "Unknown temperatureEvolution. Exiting." << endl; exit(-1); }
    // Read the data
    getline(inputFile, line); // read number data points
    istringstream ss(line);
    ss >> nPts;
    if (nPts>0) {
    	fnct = new double[nPts];
    	fncT = new double[nPts];
    	fncax = new double[nPts];
    	fncay = new double[nPts];
    	fncaz = new double[nPts];
    	fncLam = new double[nPts];
    
    	int i = 0;
    	while (getline(inputFile, line))
    	{
        	istringstream ss(line);
        	ss >> fnct[i] >> fncT[i] >> fncax[i] >> fncay[i] >> fncaz[i] >> fncLam[i];
        	i++;
    	}
    	inputFile.close();
    	// load range of interpolation
    	lxmin = log(fnct[0]);  // take log here and below because we are interpolating on log-log grid
    	lxmax = log(fnct[nPts-1]);
    }
}

void loadInterpolation(string filename, vector<string> *metadata) {
    readData(filename,metadata);

    if (nPts>1) {
    // convert to log-log
    for (int i=0; i<nPts; i++) {
        fnct[i] = log(fnct[i]);
        fncT[i] = log(fncT[i]+1e-04);
        fncax[i] = log(fncax[i]);
        fncay[i] = log(fncay[i]);
        fncaz[i] = log(fncaz[i]);
        fncLam[i] = log(fncLam[i]+1e-04);
    }
    acc_T = gsl_interp_accel_alloc();
    acc_ax = gsl_interp_accel_alloc();
    acc_ay = gsl_interp_accel_alloc();
    acc_az = gsl_interp_accel_alloc();
    acc_Lam = gsl_interp_accel_alloc();
    if (nPts>=3) {
        spline_T = gsl_spline_alloc(gsl_interp_cspline, nPts);
        spline_ax = gsl_spline_alloc(gsl_interp_cspline, nPts);
        spline_ay = gsl_spline_alloc(gsl_interp_cspline, nPts);
        spline_az = gsl_spline_alloc(gsl_interp_cspline, nPts);
        spline_Lam = gsl_spline_alloc(gsl_interp_cspline, nPts);
    } else if (nPts==2) {
        spline_T = gsl_spline_alloc(gsl_interp_linear, nPts);
        spline_ax = gsl_spline_alloc(gsl_interp_linear, nPts);
        spline_ay = gsl_spline_alloc(gsl_interp_linear, nPts);
        spline_az = gsl_spline_alloc(gsl_interp_linear, nPts);
        spline_Lam = gsl_spline_alloc(gsl_interp_linear, nPts);
    }
    gsl_spline_init(spline_T, fnct, fncT, nPts);
    gsl_spline_init(spline_ax, fnct, fncax, nPts);
    gsl_spline_init(spline_ay, fnct, fncay, nPts);
    gsl_spline_init(spline_az, fnct, fncaz, nPts);
    gsl_spline_init(spline_Lam, fnct, fncLam, nPts);
    }
}

void freeInterpolation() {
    if (nPts>1) {
        gsl_spline_free(spline_T);
        gsl_interp_accel_free(acc_T);
        
        gsl_spline_free(spline_ax);
        gsl_interp_accel_free(acc_ax);
        
        gsl_spline_free(spline_ay);
        gsl_interp_accel_free(acc_ay);
        
        gsl_spline_free(spline_az);
        gsl_interp_accel_free(acc_az);
        
        gsl_spline_free(spline_Lam);
        gsl_interp_accel_free(acc_Lam);
    }
}

double interpolateT(double x) {
    double lx = log(x);
    if (nPts==0 || nPts==1) return 0;
    if (lx < lxmin) {
        return 0;
    }
    else if (lx > lxmax) {
        return 0;
    }
    else return exp(gsl_spline_eval(spline_T, lx, acc_T));
}

double interpolateax(double x) {
    double lx = log(x);
    if (nPts==0 || nPts==1) return 1;
    if (lx < lxmin) {
        return 1;
    }
    else if (lx > lxmax) {
        return 1;
    }
    else return exp(gsl_spline_eval(spline_ax, lx, acc_ax));
}

double interpolateay(double x) {
    double lx = log(x);
    if (nPts==0 || nPts==1) return 1;
    if (lx < lxmin) {
        return 1;
    }
    else if (lx > lxmax) {
        return 1;
    }
    else return exp(gsl_spline_eval(spline_ay, lx, acc_ay));
}

double interpolateaz(double x) {
    double lx = log(x);
    if (nPts==0 || nPts==1) return 1;
    if (lx < lxmin) {
        return 1;
    }
    else if (lx > lxmax) {
        return 1;
    }
    else return exp(gsl_spline_eval(spline_az, lx, acc_az));
}

double interpolateLam(double x) {
    double lx = log(x);
    if (nPts==0 || nPts==1) return 0;
    if (lx < lxmin) {
        return 0;
    }
    else if (lx > lxmax) {
        return 0;
    }
    else return exp(gsl_spline_eval(spline_Lam, lx, acc_Lam));
}
