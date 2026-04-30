/*
 
 outputroutines.h
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
*/

#include <vector>

#ifndef __outputroutines_h__
#define __outputroutines_h__

using namespace std;

const int dwidth = 10;

int createOutputDirectory(int addSeed, long unsigned int seed); 
void outputSnapshot(dcomp* wf, int iter);
void outputSummary(int step, double norm, int cstate, int lval, double rexp, double evac, double* overlaps, int nstates, bool jump);
void outputRatios(vector<string> *metadata, int nstates, double* overlaps, double* overlaps0);
void outputBasisFunctions();
void outputSummaryFileDelimiter(int step);

void print_header(int nstates);
void print_line();

extern char outputDirName[255];

#endif /* __outputroutines_h__ */
