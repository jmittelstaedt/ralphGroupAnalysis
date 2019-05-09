import os
import os.path

import numpy as np

from .constants import *
from .parsers import FMR_parser
from .converters import FMRConverter
from .baseAnalysis import baseAnalysis, parse_series_file
from .dataset_manipulation import fit_dataset, plot_dataset, analyzedFit

def kittel_oop(H, gamma, Meff, Ha=0):
    """ Kittel formula (out-of-plane) as a function of field, returns frequency in GHz
    H: Applied field (Oe)
    gamma: Gyromagnetic ratio (GHz/Oe)
    Meff: Effective magnetization (emu/cc)
    Ha: Anisotropy field (Oe)
    """
    return (gamma/(2.*np.pi))*np.sqrt(H + Ha - 4*np.pi*Meff)

def inverted_kittel_oop(f, gamma,Meff,Ha=0):
    """ Kittel formula (out-of-plane) as a function of frequency, returns field in Oe
    f: Frequency (GHz)
    gamma: Gyromagnetic ratio (GHz/Oe)
    Meff: Effective magnetization (emu/cc)
    Ha: Anisotropy field (Oe)
    """
    return -Ha + 4*np.pi*Meff + 2*np.pi*f/gamma

class FMRAnalysis(baseAnalysis):

    BFIELD_DIM = 'field_strength'
    X_DATA_VAR = 'X (V)' # QUESTION: Will this be a problem?

    def __init__(self, sweep_type="analog"):
        super().__init__()
        self.procedure_swept_col = self.BFIELD_DIM # Shouldn't actually use this.
        self.parser = FMR_parser
        self.codename_converter = FMRConverter
        self.series_swept_params = ['frequency']

    def plot_resonances(self, **kwargs):
        """
        Plots just the resonances.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~dataset_manipulation.plot_dataset`
        """

        plot_dataset(self.sweep_ds, self.BFIELD_DIM, self.X_DATA_VAR, **kwargs)

    @staticmethod
    def lorentzian_derivative_guess(H, X, **kwargs):
        """
        Guessing for parameters in the lorentzian derivative, for use with
        fit_dataset
        """
        amplitude_guess = np.abs(X.max() - X.min())

        field_at_max = H[X.argmax()]
        field_at_min = H[X.argmin()]
        resonance_guess = (field_at_max + field_at_min)/2
        peak_to_peak = field_at_min - field_at_max
        linewidth_guess = np.abs(np.sqrt(3) * peak_to_peak)
        # print([amplitude_guess, resonance_guess, linewidth_guess, 0.0])
        return np.array([amplitude_guess, resonance_guess, linewidth_guess, 0.0])

    @staticmethod
    def lorentzian_derivative(H, A, Hres, delta_H, C):
        """ Lorentzian derviative fitting function
        H: Field (Oe)
        A: Amplitude (a.u.)
        Hres: Lorentzian Peak Extrema (Oe)
        delta_H: Full Width at Half Maximum (Oe)
         """
        return -2.*A*(H-Hres)/((delta_H/2)**2.*(1.+(H-Hres)**2./(delta_H/2)**2.)**2.) + C

    def fit_resonances(self, **kwargs):
        """
        Fits the resonances
        """

        lobounds = [0, -np.inf, 0, -np.inf]
        upbounds = np.inf

        afit  = fit_dataset(self.sweep_ds, FMRAnalysis.lorentzian_derivative,
                           FMRAnalysis.lorentzian_derivative_guess,
                           ['A','B0','Delta', 'C'], self.BFIELD_DIM, self.X_DATA_VAR,
                           bounds = (lobounds, upbounds), **kwargs)

        return FMRResonanceFit.from_analyzedFit(afit)


class FMRResonanceFit(analyzedFit):
    """
    Class to contain methods related to FMR resonance fit parameters. Wrapper
    around :class:`~baseAnalysis.analyzedFit`
    """

    FREQUENCY_DIM = 'frequency'

    @staticmethod
    def from_analyzedFit(afit):
        """
        Returns an instance of this class starting from an analyzedFit instance

        Parameters
        ----------
        afit : analysis.baseAnalysis.analyzedFit instance
        """
        return FMRResonanceFit(afit.fit_ds, afit.main_ds, afit.fit_func,
                            afit.guess_func, afit.param_names, afit.xname,
                            afit.yname, afit.yerr_name)

    # TODO: would be nice to have this use the gamma fit to
    # by the resonance fitting.
    @staticmethod
    def linear_linewidth(f, alpha, delta_H0):
        """ Linear form of the linewidth as a function of frequency
        f : float
            Frequency (GHz)
        alpha : float
            Damping
        delta_H0 : float
            Inhomogeneous linewidth (Oe)
        """
        return f*alpha/(0.0028) + delta_H0

    @staticmethod
    def linear_linewidth_guess(f, Delta, **kwargs):
        return [0.01, 1]

    def fit_linewidth(self, **kwargs):
        """
        Fits the linewidth data
        """
        lobounds = [0,0]
        upbounds = np.inf
        return self.fit_params(FMRResonanceFit.linear_linewidth,
                               FMRResonanceFit.linear_linewidth_guess,
                               ['alpha', 'DeltaH0'],
                               self.FREQUENCY_DIM, 'Delta',
                               bounds = (lobounds, upbounds),
                               **kwargs)

    @staticmethod
    def inverted_kittel_easy(f, Meff, Ha=0):
        """ Kittel formula as a function of frequency, returns field in Oe.
            Assumes anisotropy field is PARALLEL to applied field.

        Parameters
        ----------
        f : float
            Frequency (GHz)
        Meff : float
            Effective magnetization (emu/cc)
        Ha : float
            Anisotropy field (Oe)
        """
        return -2*np.pi*(Meff) + np.sqrt((2*np.pi*Meff)**2+f**2/0.0028**2) - Ha

    @staticmethod
    def inverted_kittel_easy_guess(f, B0, **kwargs):
        """
        Generates a guess for the parameters of the inverted Kittel function
        """
        return np.array([1e4, 1])

    @staticmethod
    def inverted_kittel_hard(f, Meff, Ha=0):
        """ Kittel formula as a function of frequency, returns field in Oe.
            Assumes anisotropy field is PERPENDICULAR to applied field.

        Parameters
        ----------
        f : float
            Frequency (GHz)
        Meff : float
            Effective magnetization (emu/cc)
        Ha : float
            Anisotropy field (Oe)
        """
        K = Ha/2*Meff
        return K/Meff - 2*Meff*np.pi + np.sqrt((K+2*Meff**2*np.pi)**2*0.0028**2 + (Meff*f)**2)/\
                                               (Meff*0.0028)

    @staticmethod
    def inverted_kittel_hard_guess(f, B0, **kwargs):
        """
        Generates a guess for the parameters of the inverted Kittel function
        """
        return np.array([1e3, 10])

    def fit_resonance(self, Ha_dir = 'easy', **kwargs):
        """
        Fits the resonant field to the Kittel formula

        Parameters
        ----------
        Ha_dir : string
            The direction of the anisotropy field with respect to the applied
            field direction.
            Accepts: 'easy' or 'hard'
        """
        if Ha_dir == 'easy':
            lobounds = [0,0]
            upbounds = [np.inf,np.inf]
            return self.fit_params(FMRResonanceFit.inverted_kittel_easy,
                                   FMRResonanceFit.inverted_kittel_easy_guess,
                                   ['Meff', 'Ha'],
                                   self.FREQUENCY_DIM, 'B0',
                                   bounds = (lobounds, upbounds),
                                   **kwargs)

        if Ha_dir == 'hard':
            lobounds = [0,0]
            upbounds = [np.inf,np.inf]
            return self.fit_params(FMRResonanceFit.inverted_kittel_hard,
                                   FMRResonanceFit.inverted_kittel_hard_guess,
                                   ['Meff', 'Ha'],
                                   self.FREQUENCY_DIM, 'B0',
                                   bounds = (lobounds, upbounds),
                                   **kwargs)
    @staticmethod
    def inverted_kittel_gen(f, Meff, phi, Ha=0):
        """ Kittel formula as a function of frequency, returns field in Oe.
            Assumes anisotropy field is PERPENDICULAR to applied field.

        Parameters
        ----------
        f : float
            Frequency (GHz)
        Meff : float
            Effective magnetization (emu/cc)
        Ha : float
            Anisotropy field (Oe)
        """
        K = Ha/2*Meff # for simplicity
        phi = np.pi/180*phi - np.pi/2 # angle between H and Ha
        M = Meff # short
        return 1/2/M * (-K + 3*K*np.cos(2*phi) - 4*M**2*np.pi +\
                        M*np.sqrt((K+4*M**2*np.pi)**2/M**2 + 4*f**2/0.0028**2 +\
                          K*np.cos(2*phi)*(2*K+8*M**2*np.pi+K*np.cos(2*phi))/M**2))

    @staticmethod
    def inverted_kittel_gen_guess(f, B0, **kwargs):
        """
        Generates a guess for the parameters of the inverted Kittel function
        """
        return np.array([1e3, 45, 100])

    def fit_resonance_gen(self, **kwargs):
        """
        Fits the resonant field to the Kittel formula

        Parameters
        ----------
        Ha_dir : string
            The direction of the anisotropy field with respect to the applied
            field direction.
            Accepts: 'parallel' or 'perpendicular'
        """
        lobounds = [0,-90,0]
        upbounds = [np.inf,90,200]
        return self.fit_params(FMRResonanceFit.inverted_kittel_gen,
                               FMRResonanceFit.inverted_kittel_gen_guess,
                               ['Meff', 'phi', 'Ha'],
                               self.FREQUENCY_DIM, 'B0',
                               bounds = (lobounds, upbounds),
                               **kwargs)
