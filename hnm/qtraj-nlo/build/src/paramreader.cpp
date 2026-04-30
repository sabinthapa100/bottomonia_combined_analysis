/*
 
 paramreader.cpp
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <string.h>

#include "trajectory.h"
#include "outputroutines.h"

using namespace std;

// this workhorse examines a key to see if it corresponds to a var we are setting
// and then attempts to set the var corresponding to key by converting value to the
// appropriate type.  lots of hardcoding here
void setParameter(const char *key, const char *value) {
    // integer params
    if (strcmp(key,"nTrajectories")==0) nTrajectories=atoi(value);
    if (strcmp(key,"num")==0) num=atoi(value);
    if (strcmp(key,"maxSteps")==0) maxSteps=atoi(value);
    if (strcmp(key,"snapFreq")==0) snapFreq=atoi(value);
    if (strcmp(key,"snapPts")==0) snapPts=atoi(value);
    if (strcmp(key,"potential")==0) potential=atoi(value);
    if (strcmp(key,"initType")==0) initType=atoi(value);
    if (strcmp(key,"initN")==0) initN=atoi(value);
    if (strcmp(key,"initL")==0) initL=atoi(value);
    if (strcmp(key,"initC")==0) initC=atoi(value);
    if (strcmp(key,"projType")==0) projType=atoi(value);
    if (strcmp(key,"nThreads")==0) nThreads=atoi(value);
    if (strcmp(key,"derivType")==0) derivType=atoi(value);
    if (strcmp(key,"nBasis")==0) nBasis=atoi(value);
    if (strcmp(key,"temperatureEvolution")==0) temperatureEvolution=atoi(value);
    if (strcmp(key,"dirnameWithSeed")==0) dirnameWithSeed=atoi(value);
    if (strcmp(key,"doJumps")==0) doJumps=atoi(value);
    if (strcmp(key,"maxJumps")==0) maxJumps=atoi(value);
    if (strcmp(key,"saveWavefunctions")==0) saveWavefunctions=atoi(value);
    if (strcmp(key,"outputSummaryFile")==0) outputSummaryFile=atoi(value);
    if (strcmp(key,"stepper")==0) stepper=atoi(value);
    // long unsigned int params
    if (strcmp(key,"randomseed")==0) randomseed=strtoul(value, NULL, 0);
    // double/float params
    if (strcmp(key,"L")==0) L=atof(value);
    if (strcmp(key,"m")==0) m=atof(value);
    if (strcmp(key,"dt")==0) dt=atof(value);
    if (strcmp(key,"alpha")==0) alpha=atof(value);
    if (strcmp(key,"T0")==0) T0=atof(value);
    if (strcmp(key,"Tf")==0) Tf=atof(value);
    if (strcmp(key,"t0")==0) t0=atof(value);
    if (strcmp(key,"tmed")==0) tmed=atof(value);
    if (strcmp(key,"kappa")==0) kappa=atof(value);
    if (strcmp(key,"gam")==0) gam=atof(value);
    if (strcmp(key,"initWidth")==0) initWidth=atof(value);
    if (strcmp(key,"rMax")==0) rMax=atof(value);
    if (strcmp(key,"mdfac")==0) mdfac=atof(value);
    // string params
    if (strcmp(key,"temperatureFile")==0) temperatureFile = string(value);
    if (strcmp(key,"basisFunctionsFile")==0) basisFunctionsFile = string(value);
    if (strcmp(key,"outfileID")==0) outfileID = string(value);
    return;
}

//
// This routine assumes that parameters are in text file with
// each parameter on a new line in the format
//
// PARAMKEY	PARAMVALUE
//
// The PARAMKEY must begin the line and only tabs and spaces
// can appear between the PARAMKEY and PARAMVALUE.
//
// Lines which begin with 'commentmarker' defined below are ignored
//
void readParametersFromFile(string filename, int echo) {
    
    string commentmarker = "//";
    char space = ' ';
    char tab = '\t';
    
    int maxline = 128; // maximum line length used in the buffer for reading
    char buffer[maxline];
    ifstream paramFile(filename.c_str());
    
    while(!paramFile.eof()) {
        paramFile.getline(buffer,maxline,'\n');
        string line = buffer; int length = strlen(buffer);
        if (line.substr(0,commentmarker.length())!=commentmarker && line.length()>0) {
            char key[64]="",value[64]="";
            int founddelim=0;
            for (int i=0;i<length;i++) {
                if (buffer[i]==space || buffer[i]==tab) founddelim=1;
                else {
                    if (founddelim==0) key[strlen(key)] = buffer[i];
                    else value[strlen(value)] = buffer[i];
                }
            }
            if (strlen(key)>0 && strlen(value)>0) {
                setParameter(key,value);
                if (echo) cout << key << " = " << value << endl;
            }
        }
    }
    
    return;
}

//
// Read parameters from commandline
//
void readParametersFromCommandLine(int argc, char** argv, int echo) {
    int optind = 1;
    while (optind < argc)
    {
        if (argv[optind][0]=='-') {
            string key = argv[optind];
            key = key.substr(1,key.length()); // remove '-'
            string value = argv[optind+1]; // load value
            if (echo) cout << key << " = " << value << endl;
            setParameter(key.c_str(),value.c_str());
            optind++;
        }
        optind++;
    }
    return;
}

//
// parameter processing prior to run
//
void processParameters() {
        
    dx = L/((double)num+1);
    dk = M_PI/L;
    mem_size_dcomp = sizeof(dcomp) * num;
    
    // check 1
    if (Tf >= T0) {
        cout << "==> Error: Tf must be lower than T0.  Exiting." << endl;
        exit(-1);
    }
    
    // check 2
    if (snapPts>num) {
        cout << "==> snapPts can't be larger than num; setting to num." << endl;
        snapPts=num;
    }
    
    // set random seed based on high res timer
    if (randomseed == 0) {
        randomseed = static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        print_line();
        cout << "==> High-res time-based seed used : " << randomseed << endl;
    }
    
    if (doJumps==1 && potential!=0) {
        cout << "==> KSU potentials do not support jumps; turning jumps off." << endl;
        doJumps=0;
        nTrajectories=1;
    }
    
    if (doJumps==0 && nTrajectories!=1) {
        cout << "==> Changed nTrajectories to 1 since jumps are turned off." << endl;
        nTrajectories=1;
    }

#if LIGHTWEIGHT == 1
    if (initType==100) {
	cout << "initType=100 is not supported in lightweight mode.  Exiting." << endl;
	exit(-1);
    }
    if (projType==1) {
	cout << "projType=1 is not supported in lightweight mode.  Exiting." << endl;
	exit(-1);
    }
#endif

}
