import corner, bilby, sys, os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import matplotlib.colors as colors
from UKF_Optim import KalmanFilterUpdateUKFOPtim
from sample import KalmanLikelihood
from corner import corner


plt.rcParams.update({'font.size': 10})
plt.ticklabel_format(useOffset=False)
'''
Enter details here.
'''
def PerformUKF():

    print("Number of processors: ", mp.cpu_count())

    # fiducial NS parameters. All units are cgs
    G=6.67e-8
    M=2.8e33
    # M = 1.77*1.989e+33
    R=1e6
    I = 8.4e44

    fID = '4u1626_9'

    Output = 'Results'

    OutputDirectory = Output+'/'+fID

    if not os.path.exists(OutputDirectory):
        os.makedirs(OutputDirectory)

    ObsDataFile = 'Data/4u1626Data.csv'
    Obsdata = pd.read_csv(ObsDataFile,sep=',', index_col=0)

    Obsdata = Obsdata.tail(30)
    Obsdata = Obsdata.reset_index(drop=True)

    plt.figure(0)
    plt.plot(Obsdata['Time'], Obsdata['F0'])
    plt.show()

    ObsScaled = np.zeros((len(Obsdata['F0']), 2))
    ObsScaled[0,0] = 0.

    for i in range(0, len(Obsdata['Time'])-1):
        ObsScaled[i+1,0] = Obsdata['Time'][i+1]-Obsdata['Time'][i]

    F0 = Obsdata['F0'].mean()
    ObsScaled[:,1] = Obsdata['F0']/F0

    omg0 = 2*np.pi*F0

    PF = Obsdata['PulsedFlux']


    MeasR = np.zeros((len(Obsdata['F0']),1))
    MeasR = Obsdata['F0Err']/F0
    MeasR = MeasR**2

    gammas = []
    sigmas = [] 

    nstates = 3
    tinit = 0.

    model = KalmanFilterUpdateUKFOPtim(Obsdata, ObsScaled, MeasR, tinit, nstates, gammas, sigmas)
    likelihood = KalmanLikelihood(model)

    # Set priors for recovery
    priors = bilby.core.prior.PriorDict()
    priors['gamma_Q'] = bilby.core.prior.LogUniform(1e-8, 1e-5, 'gamma_Q')
    priors['gamma_S'] = bilby.core.prior.LogUniform(1e-8, 1e-5, 'gamma_S')
    priors['sigma_Q'] = bilby.core.prior.LogUniform(1e-6, 1e-1, 'sigma_Q')
    priors['sigma_S'] = bilby.core.prior.LogUniform(1e-6, 1e-1, 'sigma_S')
    priors['beta1'] = bilby.core.prior.LogUniform(1e-12, 1e-7, 'beta1')
    priors['beta2'] = bilby.core.prior.LogUniform(1e-12, 1e-7, 'beta2')

    # Do the parameter estimation

    result = bilby.run_sampler(likelihood, priors, 
                        sampler='dynesty', sample='rwalk', walks=10, npoints=500,
                        resume=True, outdir=OutputDirectory, npool = mp.cpu_count()-2, dlogz=0.1,
                        label=fID, check_point_plot=False, plot=True)
    
    # result = bilby.result.read_in_result(filename='{}/{}_result.json'.format(OutputDirectory,fID))
    # result = bilby.result.read_in_result(filename='Results/4U1626_8_U/4U1626_8_result.json')
    
    samples = result.posterior.to_numpy()[:, :6].copy()
    samples[:, 0:6] = np.log10(samples[:, 0:6])

    # samples_2 = result2.posterior.to_numpy()[:, :6].copy()
    # samples_2[:, 0:6] = np.log10(samples_2[:, 0:6])

    # result.posterior.sample for future
    samples2 = samples.copy()
    beta1s = 10**(samples2[:, 4])
    beta2s = 10**(samples2[:,5])

    Qbar = I*(omg0)**(4/3)*beta1s**(4/3)/((G*M)**(2/3) * (beta2s)**(1/3))
    Sbar = I*(omg0)**3*(beta1s)**3/(2**(5/2)*np.pi*G*M*(beta2s)**2)
    MagMom = 2**(-5/2)*(np.pi)**(-7/10)*(G*M)**(3/5)*Qbar**(6/5)*Sbar**(-7/10)

    print(Qbar[-1], Sbar[-1], MagMom[-1])

    Params = np.zeros((len(beta1s),3))
    Params[:,0] = Qbar
    Params[:,1] = Sbar 
    Params[:,2] = MagMom
    
    fig = corner(samples, 
                 smooth=False, 
                 smooth1d=False, 
                 color='C2', 
                 plot_datapoints=True,
                 quantiles=[0.16,0.50,0.84],
                 title_kwargs={"fontsize": 18},
                 show_titles=True)
    
    plt.savefig("{}/{}_scaled_corner.png".format(OutputDirectory, fID))
    plt.close()

    # fig = corner(samples, 
    #              smooth=False, 
    #              smooth1d=False, 
    #              color='k',
    #             #  plot_datapoints=True,
    #              quantiles=[0.16,0.50,0.84],
    #              title_kwargs={"fontsize": 18},
    #              show_titles=True)
    # corner(samples_2, color='g', fig=fig)
    
    # plt.savefig("{}/{}_CombinedScaled_corner.png".format(OutputDirectory, fID))
    # plt.close()

    # truths=[0.0,0.0,3.83e30]

    # Prange = [(1e16,4e16),(2e6,5e6),(2e29,5e30)]

    # plt.figure(10)
    # # plt.hist(MagMom,bins=40, range = (1e30,2e31))
    # plt.show()

    fig = corner(Params, 
                 smooth=False, 
                 smooth1d=False,
                #  truths=truths,
                #  range=Prange,
                 color='C2')                 
                #  plot_datapoints=True,
                #  quantiles=[0.16,0.50,0.84],
                #  title_kwargs={"fontsize": 18},
                #  show_titles=True)
    
    plt.savefig("{}/{}_Params.png".format(OutputDirectory, fID))
    plt.close()

    param_dict={'gamma_Q': 10**samples[-1, 0], 
                'gamma_S':  10**samples[-1, 1], 
                'sigma_Q': 10**samples[-1, 2], 
                'sigma_S':  10**samples[-1, 3], 
                'beta1': 10**samples[-1, 4], 
                'beta2': 10**samples[-1, 5]}


    Obsdata['Time']/=86400
    print(param_dict)
    #param_dict = {'a':a, 'b':b, 'sigma2': sigma**2, 'x0': x0}
    #Try the Kalman filter with the best fit parameters.
    xx, px, ll = model.ll_on_data(params = param_dict, returnstates = True)

    df_TS = pd.DataFrame({'Time': Obsdata['Time'],
                          'omg': omg0*xx[:,0],
                          'Q': Qbar[-1]*xx[:,1],
                          'S': Sbar[-1]*xx[:,2],
                          'F0': F0*ObsScaled[:,1],
                          'FEst': F0*xx[:,0],
                          'Flux': PF})
    
    df_TS.to_csv(OutputDirectory+'/summary.csv')

    Rm = (2*np.pi**(2/5))**(-1)*(G*M)**(1/5)*(Qbar[-1]*xx[:,1])**(2/5)*(Sbar[-1]*xx[:,2])**(-2/5)
    Rc = (G*M)**(1/3)*(omg0*xx[:,0])**(-2/3)
    Torque = (G*M*Rm)**(1/2)*(1. - (Rm/Rc)**(3/2))*Qbar[-1]*xx[:,1]/I

    plt.figure(0)
    plt.plot(Obsdata['Time'], F0*ObsScaled[:,1])
    plt.plot(Obsdata['Time'], xx[:,0]*F0)
    plt.savefig('{}/{}_PEstimate.png'.format(OutputDirectory,fID))

    plt.figure(1)
    plt.plot(Obsdata['Time'], Qbar[-1]*xx[:,1]*(365*86400/(1000*1.989e+30)))
    plt.tight_layout()
    plt.savefig('{}/{}_QEstimate.png'.format(OutputDirectory,fID))

    plt.figure(2)
    plt.plot(Obsdata['Time'], Sbar[-1]*xx[:,2])
    plt.savefig('{}/{}_SEstimate.png'.format(OutputDirectory,fID))

    plt.figure(3)
    plt.plot(Obsdata['Time'], (Rm/Rc)**(3/2), 'c+')
    plt.axhline(y=0.6)
    plt.axhline(y=0.45)
    # plt.ylim([0,1.25])
    # plt.plot(Obsdata['Time'], Rm, 'bo')
    plt.tight_layout()
    plt.savefig('{}/{}_RadEstimate.png'.format(OutputDirectory,fID))

    plt.figure(4)
    plt.plot(Obsdata['Time'], (G*M/R)*Qbar[-1]*xx[:,1])
    plt.tight_layout()
    plt.savefig('{}/{}_LEstimate.png'.format(OutputDirectory,fID))

    plt.figure(5)
    plt.scatter(PF, (Rm/Rc)**(3/2), c=Obsdata['Time'], cmap='seismic', s=75)
    cbar = plt.colorbar()
    cbar.set_label('Time')
    plt.tight_layout()
    plt.savefig('{}/{}_Correlation.png'.format(OutputDirectory,fID))

    plt.figure(6)
    plt.scatter(PF, Torque, c=Obsdata['Time'], cmap='seismic', s=75)
    cbar = plt.colorbar()
    cbar.set_label('Time')
    plt.tight_layout()
    plt.savefig('{}/{}_TorqueLumCorr.png'.format(OutputDirectory,fID))

    plt.figure(7)
    plt.plot(Obsdata['Time'], Torque/np.abs(Torque.max()),  'o')
    # plt.plot(Obsdata['Time'], (Rm/Rc)**(3/2),  'o')
    # plt.plot(Obsdata['Time'],  ObsScaled[:,1]/ ObsScaled[:,1].max(),  'o')
    plt.tight_layout()
    plt.savefig('{}/{}_Torque.png'.format(OutputDirectory,fID))

    plt.figure(8)
    plt.plot(Obsdata['Time'], Rm,  'mo', alpha=0.2, markeredgecolor='k')
    plt.plot(Obsdata['Time'], Rc,  'co', alpha=0.2, markeredgecolor='k')
    # plt.plot(Obsdata['Time'],  ObsScaled[:,1]/ ObsScaled[:,1].max(),  'o')
    plt.tight_layout()
    plt.savefig('{}/{}_Radii.png'.format(OutputDirectory,fID))

if __name__=="__main__":
    PerformUKF()