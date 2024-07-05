import numpy as np
import bilby

class KalmanLikelihood(bilby.Likelihood):
    def __init__(self, NSmodel, parameters=None):
        if parameters is None:
            parameters={'gamma_omega': None, 'gamma_Q': None, 'gamma_S': None, 'gamma_eta': None,
                        'sigma_Q': None, 'sigma_S': None, 'sigma_eta': None,
                        'EFACP': None, 'EQUADP': None, 'EFACL': None, 'EQUADL': None, 'beta': None,
                        'DeltaR2': None, 'Rmstar': None, 'Rin': None, 'beta1': None, 'beta2': None,
                        'lambda1': None, "lambda2": None, "lambda3": None, 'lambda4': None,
                        'chi1': None, 'chi2': None, 'chi3': None, 'chi4': None, 'chi5': None, 
                        'gamma_OQ': None, 'gamma_OS': None, 'QParam': None, 'RParam': None}

            
        super().__init__(parameters=parameters)
        self.NSmodel = NSmodel

    def log_likelihood(self):
        try:
            ll = self.NSmodel.ll_on_data(self.parameters)
        except (np.linalg.LinAlgError, RuntimeWarning, ValueError, RuntimeError):
            ll= -np.inf
        if np.isnan(ll):
            ll = -np.inf
        return ll
