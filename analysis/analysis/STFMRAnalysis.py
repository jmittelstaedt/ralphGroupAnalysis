import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit
from scipy.signal import savgol_filter
import pandas as pd

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis, analyzedFit, parse_series_file, plot_dataset
from .baseAnalysis import get_coord_selection, fit_dataset, load_procedure_files
from ..procedures import STFMRProcedure, STFMRCryoProcedure
# from .constants import *

class STFMRAnalysis(baseAnalysis):
    """
    Class to contain all STFMR related functions, and acts as a convenient
    container for importing and storing datasets etc.
    Note that:
    1. this is fairly general, so if you end up sweeping over more stuff
    than before, it should be able to handle that transparently.
    2. this is written so that if names of parameters or data columns
    change, we just need to modify some class variables and all of the
    analysis should still work.

    Parameters
    ----------
    swept_temp : bool
        Whether temperature was swept in this scan and hence should be a
        dimension of :attr:`.STFMRAnalysis.sweep_ds`

    Attributes
    ----------
    sweep_ds : xarray.Dataset
        Dataset containing the data
    procedure_swept_col : str
        column swept in the procedure
    series_swept_params : list of str
        parameters swept in the series
    gamma : float
        Gyromagnetic Ratio in GHz/T
    mu0 : float
        Permeability of Free Space
    hbar : float
        Planck's Reduced Constant
    echarge : float
        magnitude of electron charnge
    muB : float
        Bohr Magneton
    procedure : pymeasure.experiment.Procedure
        The procedure class which created the data files. Used for importing
        using PyMeasure
    """

    # PyMeasure procedure names of parameters, for accessing things from
    # xarray Datasets
    BFIELD_DIM = 'field_strength'
    ANGLE_DIM = 'field_azimuth'
    FREQUENCY_DIM = 'rf_freq'
    TEMPERATURE_DIM = 'temperature'
    X_DATA_VAR = 'X'

    def __init__(self, swept_temp = False):
        """
        Instantiates the analysis object with an empty dataset with the correct
        dimension names. Records the guesses for initial parameters for
        fitting of the resonances.

        Must load data with a separate method.
        """

        super().__init__()
        self.procedure_swept_col = self.BFIELD_DIM
        self.series_swept_params = [self.ANGLE_DIM, self.FREQUENCY_DIM]
        self.procedue = STFMRProcedure

        if swept_temp:
            self.series_swept_params.append(self.TEMPERATURE_DIM)
            self.procedue = STFMRCryoProcedure

        self.fit_ds = None

    def load_old_procedures(self, direc, series_filenames = []):
        """
        Loads procedures from Neal's code and makes them work with the rest
        of the stuff in here.

        Parameters
        ----------
        direc : str
            Directory containing a bunch of the data files.
        """
        procedure_swept_col = 'Magnetic Field (T)'
        series_swept_params = ['static_angle','RFfrequency','temp_goal']
        if series_filenames == []:
            series_filenames = os.listdir(direc)


        sweep_ds = load_procedure_files(direc, series_filenames,
                                        procedure_swept_col, series_swept_params)
        sweep_ds.rename({
            'static_angle': self.ANGLE_DIM,
            'RFfrequency': self.FREQUENCY_DIM,
            'Magnetic Field (T)': self.BFIELD_DIM,
            'X Voltage (V)': self.X_DATA_VAR,
            'temp_goal': self.TEMPERATURE_DIM
            },
            inplace=True)
        self.sweep_ds = sweep_ds
        self.coords = self.sweep_ds.coords
        self.data_vars = self.sweep_ds.data_vars
        self.dims = self.sweep_ds.dims

    def separate_field_data(self, phi_offset=0., reverse=False):
        """
        Separates the positive and negative field data and assigns the negative
        field data to an angle 180 deg from the corresponding positive field
        data. Shifts the angle coordinates as well if requested.

        Parameters
        ----------
        phi_offset : float
            Angle to offset the angles by. If ``reverse`` is ``True``, we do
            ``phi_offset - phi``, otherwise ``phi - phi_offset``
        reverse : bool
            Whether to reverse the angular coordinates as well, i.e. go from
            increasing phi meaning clockwise to counter-clockwise

        Returns
        -------
        None
            Modifies :attr:`~.STFMRAnalysis.sweep_ds` in-place

        Raises
        ------
        ValueError
            If there is no negative field data. This could either mean none was
            taken or that this method has already been used.
        """
        pfield = self.sweep_ds.where(self.sweep_ds[self.BFIELD_DIM]>0, drop=True)
        nfield = self.sweep_ds.where(self.sweep_ds[self.BFIELD_DIM]<0, drop=True)

        if nfield[self.X_DATA_VAR].size == 0:
            raise ValueError('''No negative field data! This method was probably
                             already ran''')

        # Ensuring that the positive and negative fields have the same length,
        # dropping low-field data if they do not.
        plen = pfield.coords[self.BFIELD_DIM].values.size
        nlen = nfield.coords[self.BFIELD_DIM].values.size
        final_len = min(plen, nlen)
        # slicing is different since pfield is 0...max_field and nfield is
        # -max_field...0
        pfield = pfield.where(pfield[self.BFIELD_DIM] == pfield[self.BFIELD_DIM][-final_len:])
        nfield = nfield.where(nfield[self.BFIELD_DIM] == nfield[self.BFIELD_DIM][:final_len])

        # Assume positive and negative field points are (nominally) the same.
        # This is not actually true since e.g. +-.05 mA current to magnet does
        # *not* give the same magnitude of field, but it is pretty close.
        nfield.coords[self.BFIELD_DIM] = pfield.coords[self.BFIELD_DIM].values[::-1]
        nfield.coords[self.ANGLE_DIM] = 180 + nfield.coords[self.ANGLE_DIM].values
        # QUESTION: is this how we want to handle offsets?
        nfield.coords[self.ANGLE_DIM] -= phi_offset
        pfield.coords[self.ANGLE_DIM] -= phi_offset
        if reverse:
            nfield.coords[self.ANGLE_DIM] *= -1
            pfield.coords[self.ANGLE_DIM] *= -1

        self.sweep_ds = xr.concat([pfield,nfield], dim='field_azimuth')
        self.coords = self.sweep_ds.coords
        self.dims = self.sweep_ds.dims
        self.data_vars = self.sweep_ds.data_vars

    @staticmethod
    def omega0(B, Meff):
        """Resonant frequency"""
        return gamma*np.sqrt(abs(B)*(abs(B) + Meff))

    @staticmethod
    def B0(f, Meff):
        """Resonant field"""
        w = 2*np.pi*f
        return 0.5*(-Meff + np.sqrt(Meff**2 + 4*w**2/(gamma)**2))

    @staticmethod
    def Delta_formula(alpha, offset, f):
        """Expected formula for the Delta parameter in lorentzians"""
        return alpha*2*np.pi*f/gamma+offset

    @staticmethod
    def sym_lor(B, B0, Delta, S):
        """Symmetric lorentzian"""
        return S*Delta**2/((np.abs(B) - B0)**2 + Delta**2)

    @staticmethod
    def asym_lor(B, B0, Delta, A):
        """Asymmetric lorentzian"""
        return A*Delta*(np.abs(B) - B0)/((np.abs(B) - B0)**2 + Delta**2)

    @staticmethod
    def total_lor(B, B0, Delta, S, A, offset):
        """Symmetric + Asymmetric lorentzian with offset"""
        return STFMRAnalysis.sym_lor(B,B0,Delta,S) + STFMRAnalysis.asym_lor(B,B0,Delta,A) + offset

    def plot_resonances(self, **kwargs):
        """
        Plots just the resonances.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~analysis.analysis.baseAnalysis.plot_dataset`
        """

        plot_dataset(self.sweep_ds, self.BFIELD_DIM, self.X_DATA_VAR, **kwargs)

    # TODO: make static method. Need to figure out how to deal with the
    # guesses for Meff, S/A45 and alpha.
    def guess_resonance_params(self, X, field, field_azimuth, rf_freq,
                               **kwargs):
        """
        Guesses resonance parameters. For use in
        :meth:`~.STFMRAnalysis.fit_resonances`.

        Parameters
        ----------
        X : np.ndarray
            Mixing voltage data to guess parameters of
        field : np.ndarray
            Field strengths of data points to fit
        field_azimuth : float
            Azimuthal angle of field with respect to device angle
        rf_freq : float
            RF frequency of applied current
        **kwargs
            A catchall to deal with any unexpected dimensions of
            :attr:`.STFMRAnalysis.sweep_ds` which should not influence the
            guesses.

        Returns
        -------
        list
            List of guesses of the resonance parameters. Format:
            [B0, Delta, S, A, offset]
        """

        offset = X.mean()
        SAsign = np.sign(np.sin(2*(field_azimuth - 90)*deg2rad)
                         * np.cos((field_azimuth - 90)*deg2rad)) + 0.01

        return np.array([STFMRAnalysis.B0(rf_freq, self.Meff_guess),
                         STFMRAnalysis.Delta_formula(self.alpha_guess, 0, rf_freq),
                         self.S45_guess*SAsign, self.A45_guess*SAsign, offset])

    def fit_resonances(self, Meff_guess, alpha_guess, S45_guess,
                                 A45_guess, **kwargs):
        """
        Fits resonances after
        :meth:`~.STFMRAnalysis.separate_field_data`
        is ran.

        This uses ``curve_fit`` and is a thin wrapper around
        :func:`~analysis.analysis.baseAnalysis.fit_dataset`.

        Parameters
        ----------
        Meff_guess : float
            guess of Meff
        alpha_guess : float
            guess of magnet damping
        S45_guess : float
            Guess of symmetric amplitude of the lorentzian with ang(B, I)=45
        A45_guess : float
            Guess of antisymmetric amplitude of the lorentzian with ang(B, I)=45
        **kwargs
            can be:
            - names of ``dims`` of ``ds``
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of ``curve_fit``

        Returns
        -------
        resonanceFit
            Object containing the results of the fitting and convenience
            functions for dealing with the fit parameters
        """

        # TODO: check that separate_pnfield_data was ran.

        # save fit guesses
        self.Meff_guess = Meff_guess
        self.alpha_guess = alpha_guess
        self.S45_guess = S45_guess
        self.A45_guess = A45_guess

        # Bound parameters so that linewidth is always nonnegative.
        lobounds = [-np.inf, 0, -np.inf, -np.inf, -np.inf]
        upbounds = np.inf

        afit = fit_dataset(self.sweep_ds, STFMRAnalysis.total_lor,
                           self.guess_resonance_params,
                           ['B0', 'Delta', 'S', 'A', 'offset'],
                           self.BFIELD_DIM, self.X_DATA_VAR,
                           bounds = (lobounds, upbounds), **kwargs)

        return resonanceFit.from_analyzedFit(afit)

class resonanceFit(analyzedFit):
    """
    Class to contain methods related to STFMR resonance fit parameters. Wrapper
    around :class:`~analysis.analysis.baseAnalysis.analyzedFit`
    """

    ANGLE_DIM = 'field_azimuth'
    FREQUENCY_DIM = 'rf_freq'
    TEMPERATURE_DIM = 'temperature'

    @staticmethod
    def from_analyzedFit(afit):
        """
        Returns an instance of this class starting from an analyzedFit instance

        Parameters
        ----------
        afit : analysis.analysis.baseAnalysis.analyzedFit instance
        """
        return resonanceFit(afit.fit_ds, afit.main_ds, afit.fit_func,
                            afit.guess_func, afit.param_names, afit.xname,
                            afit.yname, afit.yerr_name)

    @staticmethod
    def B0(f, Meff):
        """Resonant field"""
        w = 2*np.pi*f
        return 0.5*(-Meff + np.sqrt(Meff**2 + 4*w**2/(gamma)**2))

    @staticmethod
    def B0_guess(B, f, **kwargs):
        """
        Guess function for use with fitting to find Meff. kwargs is used
        to catch any possible unexpected fit_ds dims, since these should not
        play a role in the fitting
        """
        return [1]

    @staticmethod
    def Delta_formula(alpha, offset, f):
        """Expected formula for the Delta parameter in lorentzians"""
        return alpha*2*np.pi*f/gamma+offset

    def fit_ang_dep(self, fit_func, guess_func, yname, param_names, **kwargs):
        """
        Fits angular dependence of some previously fit parameters. Thin wrapper
        around :func:`~analysis.analysis.baseAnalysis.fit_dataset`.

        Parameters
        ----------
        fit_func : function
            function to fit data to
        guess_func : function
            function to generate guesses of parameters. Arguments must be:
            - 1D numpy array of y data
            - numpy array of x data
            - keyword arguments, with the keywords being all dims of
            :attr:`~.analyzedFit.fit_ds` besides ``xname``. Values passed will be
            individual floats of coordinate values corresponding to those dims.
            All arguments must be accepted, not all must be used.
            As a hint, if designing for unexpected dims you can include **kwargs at
            the end. This will accept any keword arguments not explicitly defined
            by you and just do nothing with them.
            Must return a list of guesses to the parameters, in the order given
            in ``param_names``
        param_names : list of str
            list of fit parameter names, in the order they are returned
            by ``guess_func``
        yname : str
            the name of the ``dim`` containing data to be fit to
        **kwargs
            can be:
            - names of ``dims`` of :attr:`~.analyzedFit.fit_ds`
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of ``curve_fit``

        Returns
        -------
        analyzedFit
            Object containing all results of fitting and some convenience methods.
        """

        return self.fit_params(fit_func, guess_func, param_names,
                          self.ANGLE_DIM, yname, **kwargs)

    def find_Meff(self, **kwargs):
        """
        Fits the resonat field as a function of frequency to find the effective
        magnetization assuming Kittel's formula holds.

        Parameters
        ----------
        **kwargs
            can be:
            - names of ``dims`` of ``fit_ds``
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of ``curve_fit``

        Returns
        -------
        analyzedFit
            An object showing the result of the fitting. Contains the effective
            magnetization as a function of any other dimensions.
        """

        return self.fit_params(resonanceFit.B0, resonanceFit.B0_guess, ['Meff'],
                               self.FREQUENCY_DIM, 'B0', **kwargs)

    def calculate_SHE(self, Meff, tmag, tnm):
        """
        Calculates the spin Hall efficiency assuming we only have
        "standard" torques in our system, i.e. a field-like torque from an
        Oersted field and a y AD torque.

        Parameters
        ----------
        Meff : xarray.Dataset
            Dataset of the effective magnetization which has coordinates
            commensurate with that of ``fit_ds``. Can use the ``fit_ds``
            attribute of the output of find_Meff.
        tmag : float
            thickness of the magnetic layer, in m
        tnm : float
            thickness of the normal metal, in m

        Returns
        -------
        xarray.DataArray
            A DataArray with the same dimensions and coordinates as ``fit_ds``
            containing the calculated SHE.
        """

        SHE = self.fit_ds['S']/self.fit_ds['A']*echarge*mu0*Meff \
               *tmag*tnm/hbar*np.sqrt(1 + Meff/self.fit_ds['B0'])
        SHE = SHE.rename('SHE')
        return SHE
