/*
 
 potential.cpp
 
 Copyright (c) Michael Strickland and Ajaharul Islam
 
 GNU General Public License (GPLv3)
 See detailed text in license directory
 
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "math.h"

#include "trajectory.h"
#include "potential.h"

using namespace std;

//
// three loop running coupling; mu should be in GeV
//
double alphas(double mu) {
    double t = 2*log(mu/LAMBDA_MS);
    return (2*B0*B0*B0*B0*B0*B0*t*t*t - 6*B0*B1*B2*log(t) - 2*B0*B0*B0*B0*B1*t*t*log(t) - B1*B1*B1*(1-4*log(t)-5*log(t)*log(t)+2*log(t)*log(t)*log(t)) + 2*B0*B0*t*(B0*B2 - B1*B1*(1+log(t)-log(t)*log(t))))/(2*B0*B0*B0*B0*B0*B0*B0*t*t*t*t);
}

//
// real part of the Munich singlet potential
//
double V_Munich_Singlet_Re(double r, double alpha, double gam, double T) {
    return -alpha/r + 0.5*gam*T*T*T*r*r;
}

//
// imag part of the Munich singlet potential
//
double V_Munich_Singlet_Im(double r, double kappa, double T) {
    return -0.5*kappa*T*T*T*r*r;
}

//
// real part of the Munich octet potential
//
double V_Munich_Octet_Re(double r, double alpha, double gam, double T) {
    return 0.125*alpha/r + 0.21875*gam*T*T*T*r*r;  // 0.125 = 1/8 // 0.21875 = 7/32
}

//
// imag part of the Munich octet potential
//
double V_Munich_Octet_Im(double r, double kappa, double T) {
    return -0.21875*kappa*T*T*T*r*r; // 0.21875 = 7/32
}

//
// real part of the KMS potential
//
double V_KMS_Re(double r, double T) {
    
    double vvac;
    
    if (r<SBDISTIGEV) vvac = -alpha/r + SIGMA*r; // below string break scale -> Cornell + constant
    else vvac = -alpha/SBDISTIGEV + SIGMA*SBDISTIGEV; // above string breaking scale -> constant
    
    if (T<1e-8) return vvac;
    
    double md = mdfac*sqrt((NC/3.+NF/6.)*4*M_PI*alpha)*T;
    // internal energy based in-medium potential
    double vmed = -alpha*(1 + md*r)*exp(-md*r)/r + 2*SIGMA*(1 - exp(-md*r))/md - SIGMA*r*exp(-md*r); // internal energy
    //double vmed = -alpha*exp(-md*r)/r + SIGMA*(1 - exp(-md*r))/md; // free energy
    
    // implement vacuum string breaking in finite T case; adhoc, but reasonable implementation
    if (vmed>-alpha/SBDISTIGEV + SIGMA*SBDISTIGEV) return vvac;
    else return vmed;
}

//
// phi function
//
double phiFunction(double rhat) {
    
    // Mathematica notebook "phiFunction.nb" generates these
    
    // Medium r
    if (rhat>2 && rhat<12.5) return -0.01966616677848211 + 0.1656797247045812*rhat + 0.398355386509078*pow(rhat,2) - 0.40827740362472287*pow(rhat,3) + 0.25510019509698195*pow(rhat,4) -
        0.1285177289776274*pow(rhat,5) + 0.05599688535281334*pow(rhat,6) - 0.0214314134281535*pow(rhat,7) + 0.007214712888876006*pow(rhat,8) -
        0.002135985427218133*pow(rhat,9) + 0.0005565983540143839*pow(rhat,10) - 0.0001278529845965009*pow(rhat,11) + 0.000025930697398553292*pow(rhat,12) -
        4.649756920895965e-6*pow(rhat,13) + 7.377451199729823e-7*pow(rhat,14) - 1.0358639282960993e-7*pow(rhat,15) + 1.2863372693961407e-8*pow(rhat,16) -
        1.4107127560241569e-9*pow(rhat,17) + 1.363099825054256e-10*pow(rhat,18) - 1.1564738402328942e-11*pow(rhat,19) + 8.575010554453819e-13*pow(rhat,20) -
        5.522492473092289e-14*pow(rhat,21) + 3.0641744635664293e-15*pow(rhat,22) - 1.4492717041393907e-16*pow(rhat,23) + 5.7613050812400555e-18*pow(rhat,24) -
        1.8885794789671063e-19*pow(rhat,25) + 4.970208651836439e-21*pow(rhat,26) - 1.0094472527596208e-22*pow(rhat,27) + 1.4851439748658838e-24*pow(rhat,28) -
        1.4085654912770015e-26*pow(rhat,29) + 6.466047857723439e-29*pow(rhat,30);
    
    // zero
    if (rhat==0) return 0;
    
    // small r
    if (rhat<=2) return pow(rhat,2)*(0.25203922281060015 + 0.04853725561439335*pow(rhat,2) + 0.0022011604896296946*pow(rhat,4) + 0.0000468860800058801*pow(rhat,6) +
                                    5.868861431717502e-7*pow(rhat,8) + log(rhat)*(-0.3333333333333333 - 0.03333333333333333*pow(rhat,2) - 0.0011904761904761904*pow(rhat,4) -
                                                                                  0.00002204585537918871*pow(rhat,6) - 2.505210838544172e-7*pow(rhat,8) - 1.9270852604185937e-9*pow(rhat,10)) + 4.855454647395215e-9*pow(rhat,10));
    
    // large r
    return -(pow(rhat,-12)*(43545600 + 403200*pow(rhat,2) + 5760*pow(rhat,4) + 144*pow(rhat,6) + 8*pow(rhat,8) + 2*pow(rhat,10) - pow(rhat,12)));
}

//
// imag part of KMS potential
//
double V_KMS_Im(double r, double T) {
    if (T<1.0e-8) return 0;
    double myalpha = alpha;
#if USE_RUNNING_COUPLING==1
    if (T>0.1) {
        myalpha = 4*alphas(2*M_PI*T)/3;
        myalpha = (myalpha<alpha ? myalpha : alpha);
    }
#endif
    double md = mdfac*sqrt((NC/3.+NF/6.)*4*M_PI*alpha)*T;
    return -myalpha*T*phiFunction(md*r);
}

double anisoMDfac(double ax, double ay, double az) {
	const double ap = 0.5*(ax+ay);
	double xi = ap*ap/az/az-1;
	double res;
	if (xi<1e-4) res = 1 - 2*xi*xi/45 + 44*xi*xi*xi/945 - 8*xi*xi*xi*xi/189;
	else if (xi>0) res = (sqrt(2.)*(1 + xi)*asin(sqrt(xi/(1 + xi))))/(sqrt(xi)*sqrt(1 + xi + (pow(1 + xi,2)*asin(sqrt(xi/(1 + xi))))/sqrt(xi)));
	else { 
		xi = fabs(xi);
		res = (sqrt(2)*asinh(sqrt(xi)/sqrt(1 - xi)))/(pow(xi,0.25)*sqrt(sqrt(xi)/(1 - xi) + asinh(sqrt(xi)/sqrt(1 - xi))));
	}
	return sqrt(res);
}

//
// real part of the anisoKMS potential
//
double V_anisoKMS_Re(double r, double T, double ax, double ay, double az, double Lam) {
    double vvac;
    if (r<SBDISTIGEV) vvac = -alpha/r + SIGMA*r; // below string break scale -> Cornell + constant
    else vvac = -alpha/SBDISTIGEV + SIGMA*SBDISTIGEV; // above string breaking scale -> constant
    if (T<1e-8) return vvac;
    
    double isomd = mdfac*sqrt((NC/3.+NF/6.)*4*M_PI*alpha)*T;
    double anisomd = anisoMDfac(ax,ay,az)*isomd;
    // internal energy based in-medium potential
    double vmed = -alpha*(1 + anisomd*r)*exp(-anisomd*r)/r + 2*SIGMA*(1 - exp(-anisomd*r))/anisomd - SIGMA*r*exp(-anisomd*r); 
    // implement vacuum string breaking in finite T case; adhoc, but reasonable implementation
    if (vmed>-alpha/SBDISTIGEV + SIGMA*SBDISTIGEV) return vvac;
    else return vmed;
}

//
// imag part of anisoKMS potential
//
double V_anisoKMS_Im(double r, double T, double ax, double ay, double az, double Lam) {
    if (T<1.0e-8) return 0;
    double myalpha = alpha;
#if USE_RUNNING_COUPLING==1
    if (T>0.1) {
        myalpha = 4*alphas(2*M_PI*T)/3;
        myalpha = (myalpha<alpha ? myalpha : alpha);
    }
#endif
    double ap = 0.5*(ax+ay);
    double xi = ap*ap/az/az-1;

    double isomd = mdfac*sqrt((NC/3.+NF/6.)*4*M_PI*alpha)*T;
    return -myalpha*Lam*phiFunction(isomd*r)/sqrt(1+xi/3.);
}

dcomp V(double r, double T) {
    return V(r, T, 1, 1, 1, T);
}

double kappaT(double T) {
    const double x = T/TC;
    double myKappa;
    if (((int)floor(kappa))==-1) { // central fit
        myKappa = 1./(-0.51888965324059919 + 0.98030718982413288*sqrt(x) - 0.083409768363912536*x + 0.0039726666253277474*pow(x,1.5));
    } else if (((int)floor(kappa))==-2) { // lower limit fit
        myKappa = 1./(-0.8071156107961191 + 1.5560083173857689*sqrt(x) - 0.14261556412444807*x + 0.0071217431979965582*pow(x,1.5));
    } else if (((int)floor(kappa))==-3) { // upper limit fit
        myKappa = 1./(-0.38196931268763431 + 0.71513037386309615*sqrt(x) - 0.058542116432588807*x + 0.0027170443410800557*pow(x,1.5));
    } else {
        myKappa = kappa;
    }
    return myKappa;
}

dcomp V(double r, double T, double ax, double ay, double az, double Lam) {
    double myKappa;
    switch(potential) {
        case 0: // Munich
            myKappa = kappaT(T);
            if (cstate==OCTET) {
                return dcomp(V_Munich_Octet_Re(r,alpha,gam,T),V_Munich_Octet_Im(r,myKappa,T));
            }
            else if (cstate==SINGLET) {
                return dcomp(V_Munich_Singlet_Re(r,alpha,gam,T),V_Munich_Singlet_Im(r,myKappa,T));
            }
            else {
                cout << "Error:  Unknown cstate.  Exiting." << endl;
                return 0;
            }
            break;
        case 1: // KSU
            return dcomp(V_KMS_Re(r,T),V_KMS_Im(r,T));
            break;
        case 2: // anisoKSU
            return dcomp(V_anisoKMS_Re(r, T, ax, ay, az, Lam),V_anisoKMS_Im(r, T, ax, ay, az, Lam));
            break;
        default:
            cout << "Error:  Unknown potential type.  Exiting." << endl;
            exit(-1);
            break;
    }
}

dcomp Veff(double r, double T) {
    return Veff(r,T,1,1,1,T);
}

dcomp Veff(double r, double T, double ax, double ay, double az, double Lam) {
    return V(r,T,ax,ay,az,Lam) + dcomp(0.5*lval*(lval+1)/m/r/r,0);
}
