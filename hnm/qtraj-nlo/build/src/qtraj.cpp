/*
 
 qtraj.cpp
 
 Copyright (c) Michael Strickland
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ctime>

using namespace std;

#include "trajectory.h"
#include "outputroutines.h"
#include "paramreader.h"
#include "wavefunction.h"

int main(int argc, char *argv[]) {
    
    print_line(); // cosmetic
    auto starttime = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "QTraj v2.0 started: " << ctime(&starttime);
#if LIGHTWEIGHT == 1
    cout << "[lightweight mode]" << endl;
#endif
    print_line(); // cosmetic
    
    // read parameters from file and command line
    readParametersFromFile("input/params.txt",1);
    if (argc>1) {
        print_line();
        cout << "Parameters from commandline" << endl;
        print_line();
        readParametersFromCommandLine(argc,argv,1);
    }
    // perform any processing of parameters necessary
    processParameters();

    // initialize evolution
    initializeEvolution();
    
    // perform vacuum evolution (tau = 0 -> tmed); same for all trajectories
    double myNorm = evolveWavefunction(0,nmed);
    for (int i=0;i<num;i++) wfncSave[i] = wfnc[i]; // save result of vacuum evolution
    
    // do rest of the evolution
    print_line();
    cout << " No jumps" << endl;
    print_line();
    myNorm = evolveWavefunction(nmed,maxSteps-nmed);
    saveNoJumpsSingletOverlaps();
    
    if (doJumps==1) {
        
        print_line();
        cout << "==> Heff final norm is " << myNorm << endl;
        print_line();
        
        for (int i=0;i<num;i++) wfnc[i] = wfncSave[i]; // load wfnc at t=tmed
        
        // now evolve trajectories starting from tmed to final time
        for (int n=0; n<nTrajectories; n++) {
            
            print_line();
            cout << " Trajectory " << n << endl;
            print_line();

 	    outputSummaryFileDelimiter(n);
           
	    if (stepper==0)  evolveWavefunctionWithJumps(nmed,myNorm);
	    else  evolveWavefunctionWithJumpsCN(nmed,myNorm);
                        
            // perform output
            outputRatios(&metadata,nBasis,singletOverlaps,singletOverlaps0);
            
            // reset wfnc back to vacuum evolved version before next trajectory
            if (n!=nTrajectories-1) {
                lval = initL;
                cstate = initC;
                for (int i=0;i<num;i++) wfnc[i] = wfncSave[i];
            }
            
        } // end loop over trajectories
        
    } else {
        
        outputRatios(&metadata,nBasis,singletOverlaps,singletOverlaps0);
        
    }
    
    // finalize evolution
    finalizeEvolution();
    
    // print done!
    print_line();
    auto endtime = chrono::system_clock::to_time_t(chrono::system_clock::now());
    cout << "Done: " << ctime(&endtime);
    print_line(); // cosmetic
}
