from logging import raiseExceptions
from pprint import pp
from signal import pause

from pyparsing import PositionToken
from models import construct_QUKFReduced, construct_QUKF_Small_dt_limit, construct_QUKF_Diagonal, construct_QUKF_Jac
from numpy.linalg import inv, det, pinv, solve, slogdet
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt
import sdeint
import scipy.linalg as la
import scipy.optimize as optimize
import numpy as np
import torch
import torchdiffeq

from numba import jit
from scipy.integrate import odeint
from numba.types import float64, int64
from numba.typed import List


################################


################################

@jit(nopython=True)
def f(t, x, beta1, beta2, gamma_Q, gamma_S):
    DEOmega = beta1 * x[1] ** (6 / 5) * x[2] ** (-1 / 5) - beta2 * x[1] ** (9 / 5) * x[2] ** (-4 / 5) * x[0]
    DEQ = -gamma_Q * (x[1] - 1.0)
    DES = -gamma_S * (x[2] - 1.0)
    return np.array([DEOmega, DEQ, DES])

@jit(nopython=True)
def rk4_step(func, x, t, dt, *args):
    k1 = dt * func(t, x, *args)
    k2 = dt * func(t + dt / 2, x + k1 / 2, *args)
    k3 = dt * func(t + dt / 2, x + k2 / 2, *args)
    k4 = dt * func(t + dt, x + k3, *args)

    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@jit(nopython=True)
def propagate_sigma_points(SigmaPoints, params, Deltat):
    beta1, beta2, gamma_Q, gamma_S = params 

    PropagatedSigmaPoints = np.empty_like(SigmaPoints)
    for i, x in enumerate(SigmaPoints):
        PropagatedSigmaPoints[i] = rk4_step(f, x, 0, Deltat, beta1, beta2, gamma_Q, gamma_S)

    return PropagatedSigmaPoints

################################

@jit(nopython=True)
def Predict(Wc, Wm, PropagatedSigmaPoints, nstates, Q):
    Xp = np.zeros((nstates, 1))
    for i in range(len(Wm)):
        Xp += Wm[i] * PropagatedSigmaPoints[i].reshape(nstates, 1)
        
    SigmaPointDiff = np.zeros((len(PropagatedSigmaPoints), nstates))
    for i, point in enumerate(PropagatedSigmaPoints):
        SigmaPointDiff[i] = point - Xp.ravel()

    Pp = np.zeros((nstates, nstates))
    for i, diff in enumerate(SigmaPointDiff):
        Pp += Wc[i] * np.outer(diff, diff)
        
    Pp += Q

    return Xp, Pp

@jit(nopython=True)
def Update(Xp, Pp, Wc, PropagatedSigmaPoints, Observation, WeightedMeas, MeasurementDiff, RMeas):
    Inn = Observation - WeightedMeas

    SigmaPointDiff = PropagatedSigmaPoints - Xp.T

    PredMeasurementCov = np.zeros((1,1))
    for i in range(len(Wc)):
        PredMeasurementCov += Wc[i] * np.outer((MeasurementDiff.T)[i], (MeasurementDiff.T)[i])
    PredMeasurementCov += RMeas

    CrossCovariance = np.zeros((len(Xp), 1))
    for i in range(len(Wc)):
        CrossCovariance += Wc[i] * np.outer(SigmaPointDiff[i], (MeasurementDiff.T)[i])


    kalman_gain = np.linalg.solve(PredMeasurementCov.T, CrossCovariance.T).T

    X = Xp + kalman_gain*Inn

    P = Pp - np.dot(kalman_gain, np.dot(PredMeasurementCov, kalman_gain.T))

    sign, log_det = np.linalg.slogdet(PredMeasurementCov)
    ll = -0.5 * (log_det + np.dot(Inn.T, np.linalg.solve(PredMeasurementCov, Inn))
                 + np.log(2 * np.pi))

    return X, P, ll

class KalmanFilterUpdateUKFOPtim(object):
    def __init__(self, ObsRaw, ObsArray, R, tinit, nstates, gammas, sigmas):
        self.ObsRaw = ObsRaw
        self.Obs = ObsArray
        self.RMeas = R
        self.t0 = tinit
        self.nstates = nstates
        self.gammas = gammas 
        self.sigmas = sigmas
        self.ll = 0

        self.Q = np.zeros((len(self.Obs),self.nstates,self.nstates))

        self.Q_Jac = np.zeros((self.nstates,self.nstates))

        self.MeasurementNoise = np.zeros((len(self.Obs),1,1))

    def Predict(self,Q):
        self.Xp, self.Pp = Predict(self.Wc, self.Wm, self.PropagatedSigmaPoints, self.nstates, Q)

    def Update(self, Observation, MeasurementNoise):
        self.X, self.P, ll = Update(self.Xp, self.Pp, self.Wc, self.PropagatedSigmaPoints,
                             Observation, self.WeightedMeas, self.MeasurementDiff, MeasurementNoise)

        self.ll += ll

    def PropagateScaledReduced(self, params, Observation):
        Deltat = Observation
        params_tuple = (params['beta1'], params['beta2'], params['gamma_Q'], params['gamma_S'])
        self.PropagatedSigmaPoints = propagate_sigma_points(self.SigmaPoints, params_tuple, Deltat)


    def PropagateScaledPropTrans(self, gammas, params, Observation):

        gamma_Q = gammas[1]
        gamma_S = gammas[2]
        lambda1 = gammas[4]
        lambda2 = gammas[5]
        lambda3 = gammas[6]

        lambda4 = params['lambda4']

        Deltat = Observation['Deltat']

        def f(x, t):
            DEOmega = lambda1*x[1]**(6/5)*x[2]**(-1/5)*np.tanh((lambda2*x[0]**(-2/3) - lambda3*x[1]**(2/5)*x[2]**(-2/5))/lambda4)
            DEQ = -gamma_Q*(x[1] - 1.)
            DES = -gamma_S*(x[2] - 1.)
            a = np.asarray([DEOmega, DEQ, DES])
            return a 

        IntegratedStates = [odeint(f, x[0:3], [0, Deltat])[-1]  for x in self.SigmaPoints]
        states = np.zeros((9,4))
        states[:,0:3] = IntegratedStates
        states[:,3] = 0.5*(1. + np.tanh((lambda2*states[:,0]**(-2/3) - lambda3*states[:,1]**(2/5)*states[:,2]**(-2/5))/lambda4))

        self.PropagatedSigmaPoints = states

    def PropagateScaledDiskTrap(self, gammas, params, Observation):

        gamma_Q = gammas[1]
        gamma_S = gammas[2]
        chi1 = gammas[5]
        chi2 = gammas[6]
        chi5 = params['chi5']
        gamma_OQ = gammas[7]
        gamma_OS = gammas[8]
        chi6 = gammas[9]
        chi7 = gammas[10]

        # [gamma_omega, gamma_Q, gamma_S, gamma_eta, chi4, chi1, chi2, gamma_OQ, gamma_OS]

        Deltat = Observation['Deltat']

        def func1(R_guess, x):
            return 0.5*x[0]*x[1]/(x[2])*(1. - np.tanh(chi1*R_guess - chi2*x[0]**(-2/3)))

        def f(x, t):
            Rin = optimize.fixed_point(func1, 
            0.5*x[0]*x[1]/x[2], args=[x],  xtol=1e1, maxiter=5)

            DEOmega =  gamma_OQ*x[1]*Rin**(1/2) - gamma_OS*chi5*x[2]*Rin**2*(1. + np.tanh(chi5**(-1)*(chi6*Rin - chi7*x[0]**(-2/3))))
            DEQ = -gamma_Q*(x[1] - 1.)
            DES = -gamma_S*(x[2] - 1.)
            a = np.asarray([DEOmega, DEQ, DES])
            return a 

        IntegratedStates = [odeint(f, x[0:3], [0, Deltat])[-1]  for x in self.SigmaPoints]
        states = np.zeros((9,4))
        states[:,0:3] = IntegratedStates

        def func1data(R_guess, Omega_data, Q_data, S_data):
            return 0.5*Omega_data*Q_data/(S_data)*(1. - np.tanh(chi1*R_guess - chi2*Omega_data**(-2/3)))
        
        Rindata = optimize.fixed_point(func1data, 
        0.5*states[:,0]*states[:,1]/states[:,2], 
        args = (states[:,0], states[:,1], states[:,2]), 
         xtol=1e1, maxiter=5)

        Qcdata = Rindata*states[:,2]/states[:,0]

        states[:,3] = (Qcdata/states[:,1])

        self.PropagatedSigmaPoints = states

    def SigmaPointMeasurementsScaled(self, params):
        Wm = self.Weights['Wm']
        # chi4 = self.gammas[4]
        SP_Meas = np.zeros((2*self.nstates+1,2))
        for i, x in enumerate(self.PropagatedSigmaPoints):
            SP_Meas[i,0] = 1./x[0] 
            SP_Meas[i,1] = x[1]*x[3]

        self.WeightedMeas = np.einsum('i,ij->j', Wm, SP_Meas)
        # self.WeightedMeas = np.asarray(sum(Weight*Measurement for 
        #         Weight, Measurement in zip(Wm, SP_Meas)))
        
        self.MeasurementDiff = [(np.asarray(SPMeas)) - (self.WeightedMeas) for SPMeas in SP_Meas]

    def SigmaPointMeasurementsScaledReduced(self, params):

        SP_Meas = np.column_stack((self.PropagatedSigmaPoints[:, 0]))

        self.WeightedMeas = np.einsum('i,ij->j', self.Wm, SP_Meas.T)
        # self.WeightedMeas = np.asarray(sum(Weight*Measurement for 
        # Weight, Measurement in zip(self.Wm, SP_Meas)))

        self.MeasurementDiff = SP_Meas - self.WeightedMeas

    def CalculateWeights(self):
        L = self.nstates
        alpha = 1e-4
        beta =  2
        kappa = 3 - self.nstates
        kappa = 0

        # Compute sigma point weights
        lambda_ = alpha**2 * (self.nstates + kappa) - self.nstates

        self.Wm = np.concatenate(([lambda_/(self.nstates+lambda_)], 
                             (0.5/(self.nstates+lambda_))*np.ones(2*self.nstates)), axis=None)
        
        self.Wc = np.concatenate(([lambda_/(self.nstates+lambda_)+(1-alpha**2+beta)], 
                             (0.5/(self.nstates+lambda_))*np.ones(2*self.nstates)), axis=None)

        self.gamma = np.sqrt(self.nstates + lambda_)

    def CalculateSigmaPoints(self, X, P):

        epsilon = 1e-18
        Pos_definite_Check= 0.5*(P + P.T) + epsilon*np.eye(len(X))
        
        U = la.cholesky(Pos_definite_Check).T # sqrt
        sigma_points = np.zeros((2*self.nstates + 1, self.nstates))
        sigma_points[0] = X
        for i in range(self.nstates):
            sigma_points[i+1] = X + self.gamma*U[:, i]
            sigma_points[self.nstates+i+1] = X - self.gamma*U[:, i]

        self.SigmaPoints = sigma_points
                
    def Get_MeasuremenNoise(self, params):
        self.MeasurementNoise[:,0,0] = self.RMeas


    def ll_on_data(self, params, returnstates=False):
        self.ll = 0
        self.X = np.ones((self.nstates, 1))

        self.P = np.eye(self.nstates)*np.max(self.RMeas)*1e2
        NObs = len(self.ObsRaw)
        if returnstates:
            xx = np.zeros((NObs,self.nstates))
            px = np.zeros((NObs,self.nstates))

        self.CalculateWeights()

        # self.Q = construct_QUKF_Diagonal(params, self.Obs[:,0])
        self.Get_MeasuremenNoise(params)
        
        i = 0
        for step, Obs in enumerate(self.Obs):
            print(i)

            self.CalculateSigmaPoints(self.X.squeeze(), self.P)

            self.PropagateScaledReduced(params, Obs[0])
            # if (Obs[1]==0):
            #     self.Q[step,:,:] = 0.0
            #     self.Predict(self.Q[step,:,:])
            # else:
            self.Q = construct_QUKF_Jac(params, Obs[0], self.X)
            self.Predict(self.Q)

            # if (Obs[1]==0):
            #     self.X = self.Xp 
            #     self.P = self.Pp
            # else:
                
            self.CalculateSigmaPoints(self.Xp.squeeze(), self.Pp)
            self.PropagatedSigmaPoints = self.SigmaPoints

            self.SigmaPointMeasurementsScaledReduced(params)

            self.Update(Obs[1], self.MeasurementNoise[step,:,:])

            if returnstates:
                xx[i,:] = self.X.squeeze()
                px[i,:] = np.diag(self.P)
                i+=1

        if returnstates:
            return xx, px, self.ll
        else:
            return self.ll




