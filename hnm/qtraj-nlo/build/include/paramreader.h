/*

   paramreader.h

   Copyright (c) Michael Strickland

   GNU General Public License (GPLv3)
   See detailed text in license directory

*/


#ifndef __paramreader_h__
#define __paramreader_h__

void readParametersFromFile(std::string filename, int echo);
void readParametersFromCommandLine(int argc, char** argv, int echo);
void processParameters();

#endif /* __paramreader_h__ */
