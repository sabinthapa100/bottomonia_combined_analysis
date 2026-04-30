/*
 
 interpolator.h
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
*/

#ifndef __interpolator_h__
#define __interpolator_h__

using namespace std;

void loadInterpolation(string filename, vector<string> *metadata);
void freeInterpolation();
double interpolateT(double x);
double interpolateax(double x);
double interpolateay(double x);
double interpolateaz(double x);
double interpolateLam(double x);

#endif /* __interpolator_h__ */
