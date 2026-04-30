/*
 
 outputroutines.cpp
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <vector>
#include <fcntl.h>
#include <limits>

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include "trajectory.h"
#include "outputroutines.h"

using namespace std;

// used to store the output directory name
char outputDirName[255];

int directoryExists(char* dirName) {
    DIR* dir = opendir(dirName);
    if (dir) {
        closedir(dir);
        return 1;
    } else if (ENOENT == errno) {
        return 0;
    } else {
        return -1;
    }
}

int createOutputDirectory(int mode, long unsigned int seed) {
    if (mode==1) snprintf(outputDirName,255,"output-%lu",seed);
    else if (mode==2) snprintf(outputDirName,255,"output-%s",outfileID.c_str());
    else snprintf(outputDirName,255,"output");

    if (directoryExists(outputDirName)==1) {
        cout << "==> Output directory '" << outputDirName << "' already exists" << endl;
        return 1;
    }
    cout << "==> Creating output directory '" << outputDirName << "'" << endl;
    if (mkdir(outputDirName, 0777) == -1)
        return -1;
    else
        return 1;
}

void outputBasisFunctions() {
    fstream out;
    char fname[512];
    snprintf(fname,512,"%s/basisfunctions.tsv",outputDirName);
    out.open(fname, ios::out);
    cout << "==> Saving basis functions" << endl;
    out << setprecision( numeric_limits<double>::digits10+2 );
    for (int i=0;i<num*nBasis;i++)
        out << real(basisFunctions[i]) << "\t" << imag(basisFunctions[i]) << endl;
    out.close();
    return;
}

void outputSnapshot(dcomp* wf, int iter) {
    fstream out;
    char fname[512];
    snprintf(fname,512,"%s/snapshot_%d.tsv",outputDirName,iter);
    out.open(fname, ios::out);
    out << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << endl; // 0 by boundary condition
    for (int i=0;i<num;i+=num/snapPts)  {
        double r = (i+1)*dx;
        out << r << "\t" << abs(wf[i]) << "\t" << real(wf[i]) << "\t" << imag(wf[i]) << endl;
    }
    out << (num+1)*dx << "\t" << 0 << "\t" << 0 << "\t" << 0 << endl; // 0 by boundary condition
    out.close();
    return;
}

void outputSummaryFileDelimiter(int step) {
    outputFile << "# Trajectory Number " << step << endl;
}

void outputSummary(int step, double norm, int cstate, int lval, double rexp, double evac, double* overlaps, int nstates, bool jump) {
   
    // output to screen
    cout << setprecision(6);
    cout << fixed;
    cout.width(dwidth);
    cout << "t = " << (t0 + step*dt)*HBARC << " fm/c";
    cout.width(dwidth);
    cout << norm;
    cout.width(dwidth/2);
    cout << "(" << cstate << "," << lval << ")";
    cout << "   ";
    //cout << scientific;
    for (int j=0; j<nstates; j++) {
        cout.width(dwidth);
        cout << overlaps[j];
        cout << "   ";
    }
    cout.width(dwidth);
    cout << rexp;
    cout.width(dwidth+3);
    cout << evac;
    if (jump) cout << " * ";
    cout << endl;
    
    if (!jump && outputSummaryFile==1) {
        // output to summary file
        outputFile << setprecision(6);
        outputFile << fixed;
        outputFile << (t0 + step*dt)*HBARC << "\t";
        outputFile << norm << "\t";
        outputFile << cstate << "\t";
        outputFile << lval << "\t";
        outputFile << scientific;
        for (int j=0; j<nstates; j++) {
            outputFile << overlaps[j];
            outputFile << "\t";
        }
        outputFile << rexp << "\t";
        outputFile << evac;
        outputFile << endl;
    }
}

struct flock* fileLock(const short type) {
    static struct flock ret ;
    ret.l_type = type ;
    ret.l_start = 0 ;
    ret.l_whence = SEEK_SET ;
    ret.l_len = 0 ;
    ret.l_pid = getpid() ;
    return &ret ;
}

void outputRatios(vector<string> *metadata, int nstates, double* overlaps, double* overlaps0) {
    
    // used for file locking below
    struct flock fl;
    fl.l_type    = F_WRLCK;   /* Test for any lock on any part of file. */
    fl.l_start   = 0;
    fl.l_whence  = SEEK_SET;
    fl.l_len     = 0;
    
    char fname[512];
    snprintf(fname,512,"%s/ratios.tsv",outputDirName);
    int fd = open(fname, O_WRONLY|O_CREAT|O_APPEND,0666);
    if (fd<0) {
        perror(fname);
        perror("Unable to open ratios.tsv file for output");
        exit(-1);
    }
    
    ostringstream ratioOut;
    for(int i=0; i<metadata->size(); i++) {
        if (i==0) ratioOut << metadata->at(i);
        else ratioOut << "\t" << metadata->at(i);
    }
    if (metadata->size() > 0) ratioOut << endl;
    for (int i=0; i<nstates; i++) {
        if (initType==0 || initType==100) { // singlet coloumb state; do nothing
            ratioOut << overlaps[i];
            if (i<nstates-1) ratioOut << "\t";
        } else { // all other IC
            if (overlaps0[i] == 0) overlaps0[i] = 1; // do not scale if they weren't there in the first place
            ratioOut << overlaps[i]/overlaps0[i];
            if (i<nstates-1) ratioOut << "\t";
        }
    }
    ratioOut << "\t" << firstRandomNumber;
    ratioOut << "\t" << initL;
    ratioOut << endl;
    string outputString = ratioOut.str();

    // get a lock to prevent anyone writing while we write
    if (fcntl(fd, F_SETLKW, &fl) == -1) {
        perror("Error getting lock on ratios.tsv file");
        exit(-1);
    }
    
    int sz = write(fd, outputString.c_str(), outputString.size());

    // release the lock
    fl.l_type = F_UNLCK;
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        perror("Error releasing lock on ratios.tsv file");
        exit(-1);
    }
    
    close(fd);
}

void print_header(int nstates) {
    print_line();
    cout.width(18);
    cout << "t [fm/c]";
    cout.width(12);
    cout << "norm";
    cout.width(12);
    cout << "(c,l)";
    cout.width(10);
    for (int j=0; j<nstates; j++) {
        int row = floor(-0.5 + sqrt(0.25 + 2 * j));
        int triangularNumber = row * (row + 1) / 2;
        int column = j - triangularNumber;
        cout.width(dwidth);
        cout << row+1 << "," << column << " ";
    }
    cout.width(12);
    cout << "<r>/s0";
    cout.width(12);
    cout << "evac" << endl;
    print_line();
    return;
}

void print_line() {
    for (int i=0;i<147;i++) cout << "-"; cout << endl;
    return;
}

