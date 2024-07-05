import os
import bilby
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 18
})

def PlotResults(df_B, df_F, Post_B, Post_F, fID, Output):


    OutputDirectory = Output+'_'+fID+'/'

    if not os.path.exists(OutputDirectory):
        os.makedirs(OutputDirectory)

    # fiducial NS parameters. All units are cgs
    G=6.67e-8
    M=2.8e33
    # M = 1.77*1.989e+33
    R=1e6
    I = 8.4e44

    T_B = df_B['Time'] # Days [MJD]

    O_B = df_B['omg']
    Q_B = df_B['Q']
    S_B = df_B['S']

    Fr_B = df_B['F0']
    FrKF_B = df_B['FEst']
    Fl_B = df_B['Flux']

    Rm  = lambda Q, S : (2*np.pi**(2/5))**(-1)*(G*M)**(1/5)*(Q)**(2/5)*(S)**(-2/5)
    Rc  = lambda Om : (G*M)**(1/3)*(Om)**(-2/3)
    FP  = lambda Rmag, Rcrot: (Rmag/Rcrot)**(3/2)
    Tor = lambda Rmag, FPar, Q: (G*M*Rmag)**(1/2)*(1. - FPar)*Q/I

    Rm_B = Rm(Q_B, S_B)
    Rc_B = Rc(O_B)
    FP_B = FP(Rm_B, Rc_B)
    Tor_B = Tor(Rm_B, FP_B, Q_B)

    # Results Fermi

    T_F = df_F['Time'] # Days [MJD]

    O_F = df_F['omg']
    Q_F = df_F['Q']
    S_F = df_F['S']

    Fr_F = df_F['F0']
    FrKF_F = df_F['FEst']
    Fl_F = df_F['Flux']

    Rm_F = Rm(Q_F, S_F)
    Rc_F = Rc(O_F)
    FP_F = FP(Rm_F, Rc_F)
    Tor_F = Tor(Rm_F, FP_F, Q_F)

    a = 0.8
    plt.figure(0)
    plt.plot(T_B, Fr_B, 'mo', alpha = 0.1)
    plt.plot(T_F, Fr_F, 'co', alpha = 0.1)
    plt.tight_layout()
    plt.savefig('{}/{}_Observations.png'.format(OutputDirectory,fID))

    plt.figure(1)
    plt.hist(Fr_B-FrKF_B, bins = 50, color='m', alpha = 0.8, edgecolor='k')
    plt.hist(Fr_F-FrKF_F, bins = 50, color='c', alpha = 0.4, edgecolor='k')
    plt.tight_layout()
    plt.savefig('{}/{}_Residuals.png'.format(OutputDirectory,fID))

    plt.figure(2)
    plt.plot(Fl_B, Tor_B, 'mo', alpha = a, markeredgecolor='k')
    plt.plot(Fl_F, Tor_F, 'co', alpha = a, markeredgecolor='k')
    plt.xlabel('$F_{X}$')
    plt.ylabel('$\\dot{\\Omega}$', rotation=0)
    plt.tight_layout()
    plt.savefig('{}/{}_TorLumCorr.png'.format(OutputDirectory,fID))

    plt.figure(4)
    plt.plot(FP_B, Fl_B, 'mo', alpha = a, markeredgecolor='k')
    plt.plot(FP_F, Fl_F, 'co', alpha = a, markeredgecolor='k')
    plt.tight_layout()
    plt.savefig('{}/{}_FluxFPCorr.png'.format(OutputDirectory,fID))

    fig, axs = plt.subplots(2,1, figsize=(10,10))
    axs[0].plot(T_B, FP_B, 'mo', alpha = a, markeredgecolor='k', label = 'BATSE')
    axs[1].plot(T_F, FP_F, 'co', alpha = a, markeredgecolor='k', label = 'Fermi GBM')

    axs[0].set_ylabel('$\\omega$', rotation=0)
    axs[1].set_ylabel('$\\omega$', rotation=0)
    axs[1].set_xlabel('Time [MJD]')

    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('{}/{}_FP_TS.png'.format(OutputDirectory,fID))

    fig, axs = plt.subplots(3,2, figsize=(10,10))
    axs[0,0].plot(T_B, O_B, 'mo', alpha = a, markeredgecolor='k', label = 'BATSE')
    axs[0,1].plot(T_F, O_F, 'co', alpha = a, markeredgecolor='k', label = 'Fermi GBM')

    axs[1,0].plot(T_B, Q_B, 'mo', alpha = a, markeredgecolor='k', label = 'BATSE')
    # axs[1,0].set_ylim([1.85e15,1.9e15])
    axs[1,1].plot(T_F, Q_F, 'co', alpha = a, markeredgecolor='k', label = 'Fermi GBM')

    axs[2,0].plot(T_B, S_B, 'mo', alpha = a, markeredgecolor='k', label = 'BATSE')
    axs[2,1].plot(T_F, S_F, 'co', alpha = a, markeredgecolor='k', label = 'Fermi GBM')

    axs[0,0].set_ylabel('$\\Omega$', rotation=0, labelpad=10)
    axs[1,0].set_ylabel('$Q$', rotation=0, labelpad=10)
    axs[2,0].set_ylabel('$S$', rotation=0, labelpad=10)
    axs[0,1].set_ylabel('$\\Omega$', rotation=0, labelpad=10)
    axs[1,1].set_ylabel('$Q$', rotation=0, labelpad=10)
    axs[2,1].set_ylabel('$S$', rotation=0, labelpad=10)
    axs[0,0].set_xlabel('Time [MJD]')
    axs[1,0].set_xlabel('Time [MJD]')
    axs[2,0].set_xlabel('Time [MJD]')
    axs[0,1].set_xlabel('Time [MJD]')
    axs[1,1].set_xlabel('Time [MJD]')
    axs[2,1].set_xlabel('Time [MJD]')

    plt.tight_layout()
    plt.savefig('{}/{}_States.png'.format(OutputDirectory,fID))


    # Posterior plots 
    # result.posterior.sample for future

    omg0_B = 2*np.pi*np.mean(Fr_B)

    beta1s_B = 10**(Post_B[:, 4])
    beta2s_B = 10**(Post_B[:, 5])

    Qbar_B = I*(omg0_B)**(4/3)*beta1s_B**(4/3)/((G*M)**(2/3) * (beta2s_B)**(1/3))
    Sbar_B = I*(omg0_B)**3*(beta1s_B)**3/(2**(5/2)*np.pi*G*M*(beta2s_B)**2)
    MagMom_B = 2**(-5/2)*(np.pi)**(-7/10)*(G*M)**(3/5)*Qbar_B**(6/5)*Sbar_B**(-7/10)

    omg0_F = 2*np.pi*np.mean(Fr_F)

    beta1s_F = 10**(Post_F[:, 4])
    beta2s_F = 10**(Post_F[:, 5])

    Qbar_F = I*(omg0_F)**(4/3)*beta1s_F**(4/3)/((G*M)**(2/3) * (beta2s_F)**(1/3))
    Sbar_F = I*(omg0_F)**3*(beta1s_F)**3/(2**(5/2)*np.pi*G*M*(beta2s_F)**2)
    MagMom_F = 2**(-5/2)*(np.pi)**(-7/10)*(G*M)**(3/5)*Qbar_F**(6/5)*Sbar_F**(-7/10)

    # plt.figure(7)
    # plt.hist(MagMom_B,bins=35, color='m', edgecolor='k', alpha=a)
    # plt.hist(MagMom_F,bins=35, color='c', edgecolor='k', alpha=a)
    # plt.axvline(x=2.4e30/2)
    # plt.axvline(x=6.3e30/2)
    # plt.show()


ID1 = 'CenX3_13'
ID2 = 'CenX3_11'

result_B = bilby.result.read_in_result(filename=f'Results/{ID1}/{ID1}_result.json')
result_F = bilby.result.read_in_result(filename=f'Results/{ID2}/{ID2}_result.json')
    
samples_B = result_B.posterior.to_numpy()[:, :6].copy()
samples_B[:, 0:6] = np.log10(samples_B[:, 0:6])

samples_F = result_F.posterior.to_numpy()[:, :6].copy()
samples_F[:, 0:6] = np.log10(samples_F[:, 0:6])

df_B = pd.read_csv(f'Results/{ID1}/summary.csv', sep=',')
df_F = pd.read_csv(f'Results/{ID2}/summary.csv', sep=',')
Output = 'Results'
fID = 'CenX3'

PlotResults(df_B, df_F, samples_B, samples_F, fID, Output)