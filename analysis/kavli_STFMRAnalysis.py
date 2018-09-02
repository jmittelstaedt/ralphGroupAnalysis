import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit
from scipy.signal import savgol_filter
import pandas as pd

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis, analyzedFit, parse_series_file
from .baseAnalysis import get_coord_selection, fit_dataset, load_procedure_files


class kavli_STFMRAnalysis(baseAnalysis):
    """
    Class to contain all STFMR related functions, and acts as a convenient
    container for importing and storing datasets etc.
    Note that:
    1. this is fairly general, so if you end up sweeping over more stuff
    than before, it should be able to handle that transparently.
    2. this is written so that if names of parameters or data columns
    change, we just need to modify some class variables and all of the
    analysis should still work.

    Attributes
    ----------
    sweep_ds : xarray.Dataset
        Dataset containing the data
    resonance_fits : baseAnalysis.analyzedFit
        fit object
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
    """

    # PyMeasure procedure names of parameters, for accessing things from
    # xarray Datasets
    BFIELD_DIM = 'field_strength'
    ANGLE_DIM = 'field_azimuth'
    FREQUENCY_DIM = 'rf_freq'
    TEMPERATURE_DIM = 'temperature'
    X_DATA_VAR = 'X'

    gamma = 2*np.pi*28.024 #GHz*Radians/T
    mu0 = 4*np.pi*1e-7 #N/A^2 i.e. T*m/A   SI for LYFE!!!!!
    hbar = 6.626*1e-34/(2*np.pi)
    echarge = 1.602e-19
    muB = 9.27400968*10**(-24)
    rad2deg = 180.0/np.pi
    deg2rad = np.pi/180.0

    # TODO: Give each one a name for plotting results? Could be useful.
    def __init__(self):
        """
        Instantiates the analysis object with an empty dataset with the correct
        dimension names. Records the guesses for initial parameters for
        fitting of the resonances.

        Must load data with a separate method.
        """

        super().__init__()
        self.procedure_swept_col = self.BFIELD_DIM
        self.series_swept_params = [self.ANGLE_DIM, self.FREQUENCY_DIM,
                                    self.TEMPERATURE_DIM]

        self.fit_ds = None
        self.resonance_fits = None

    def load_old_procedures(self, direc):
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

    def load_sweeps(self, direc, sweep_files):
        """
        Loads data from a set of sweep files, each at a different temperature

        Puts all of the filenames together and runs regular load_sweeps
        on it.

        Parameters
        ----------
        sweep_files : list of str
            a list of sweep files
        """

        if not os.path.isdir(direc):
            raise ImportError("Given directory does not exist!")

        all_procedure_files = []
        all_swept_params = []

        for sfile in sweep_files:
            procedure_files, procedure_swept_col, \
             swept_params = parse_series_file(direc, sfile)
            all_procedure_files += procedure_files
            all_swept_params += swept_params

        all_swept_params = list(set(all_swept_params)) # make all params unique

        # ensure all expected swept params are included
        for param in self.series_swept_params:
            if param not in all_swept_params:
                all_swept_params.append(param)

        self.sweep_ds = load_procedure_files(direc, all_procedure_files,
                                                 self.procedure_swept_col,
                                                 all_swept_params)
        self.coords = self.sweep_ds.coords
        self.dims = self.sweep_ds.dims
        self.data_vars = self.sweep_ds.data_vars

    def separate_field_data(self, phi_offset=0., reverse=False):
        """
        Separates the positive and negative field data and assigns the negative
        field data to an angle 180 deg from the corresponding positive field
        data. Shifts the angle coordinates as well if requested.
        
        Parameters
        ----------
        phi_offset : float
            Angle to offset the angles by. If ``reverse`` is ``True``, we do
            phi_offset - phi, otherwise phi - phi_offset
        reverse : bool
            Whether to reverse the angular coordinates as well, i.e. go from 
            increasing phi meaning clockwise to counter-clockwise
            
        Returns
        -------
        None
            Modifies sweep_ds in place
        """
        pfield = self.sweep_ds.where(self.sweep_ds[self.BFIELD_DIM]>0, drop=True)
        nfield = self.sweep_ds.where(self.sweep_ds[self.BFIELD_DIM]<0, drop=True)
        # Assume positive and negative field points are (nominally) the same.
        # this is not too great, but w/e. Better for new procedures since
        # calibrations were inverted
        nfield.coords[self.BFIELD_DIM] = pfield.coords[self.BFIELD_DIM].values[::-1]
        nfield.coords[self.ANGLE_DIM] = 180 + nfield.coords[self.ANGLE_DIM].values
        # QUESTION: is this how we want to handle offsets?
        nfield.coords[self.BFIELD_DIM] -= phi_offset
        pfield.coords[self.BFIELD_DIM] -= phi_offset
        if reverse:
            nfield.coords[self.BFIELD_DIM] *= -1
            pfield.coords[self.BFIELD_DIM] *= -1

        self.sweep_ds = xr.concat([pfield,nfield], dim='field_azimuth')
        self.coords = self.sweep_ds.coords
        self.dims = self.sweep_ds.dims
        self.data_vars = self.sweep_ds.data_vars

    def omega0(self, B, Meff):
        """Resonant frequency"""
        return self.gamma*np.sqrt(abs(B)*(abs(B) + Meff))

    def B0(self, f, Meff):
        """Resonant field"""
        w = 2*np.pi*f
        return 0.5*(-Meff + np.sqrt(Meff**2 + 4*w**2/(self.gamma)**2))

    def Delta_formula(self, alpha, offset, f):
        """Expected formula for the Delta parameter in lorentzians"""
        return alpha*2*np.pi*f/self.gamma+offset

    def sym_lor(self, B, B0, Delta, S):
        """Symmetric lorentzian"""
        return S*Delta**2/((np.abs(B) - B0)**2 + Delta**2)

    def asym_lor(self, B, B0, Delta, A):
        """Asymmetric lorentzian"""
        return A*Delta*(np.abs(B) - B0)/((np.abs(B) - B0)**2 + Delta**2)

    def total_lor(self, B, B0, Delta, S, A, offset):
        """Symmetric + Asymmetric lorentzian with offset"""
        return self.sym_lor(B,B0,Delta,S) + self.asym_lor(B,B0,Delta,A) + offset

    def guess_resonance_params(self, X, field, field_azimuth, rf_freq,
                               temperature):
        """
        Guesses resonance parameters. For use in fit_separated_resonances.
        
        Parameters
        ----------
        X : np.ndarray
            Voltage data to guess parameters of
        field : np.ndarray
            Field strengths of data points to fit
        field_azimuth : float
            Azimuthal angle of field with respect to device angle
        rf_freq : float
            RF frequency of applied current
        temperature : float
            Temperature measurements were made at
            
        Returns
        -------
        list
            List of guesses of the resonance parameters. Format:
            [B0, Delta, S, A, offset]
        """

        offset = X.mean()
        SAsign = np.sign(np.sin(2*(field_azimuth - 90)*self.deg2rad)
                         * np.cos((field_azimuth - 90)*self.deg2rad)) + 0.01

        return np.array([self.B0(rf_freq, self.Meff_guess),
                         self.Delta_formula(self.alpha_guess, 0, rf_freq),
                         self.S45_guess*SAsign, self.A45_guess*SAsign, offset])

    def fit_resonances(self, Meff_guess, alpha_guess, S45_guess,
                                 A45_guess, **kwargs):
        """
        Fits resonances after separate_pnfield_data is ran. Saves it to
        resonance_fits attribute as an analyzedFit object. This just uses curve_fit,
        none of the fancy harsh penalties given before. Thin wrapper around
        baseAnalysis.fit_dataset.

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
        kwargs
            can be:
            - names of dims of ds (cannot include xname)
            values should eitherbe single coordinate values or lists of coordinate
            values of those dims. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of curve_fit
            
        Returns
        -------
        None
            Saves the results into resonance_fits as an analyzedFit object
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

        self.resonance_fits = fit_dataset(self.sweep_ds, self.total_lor,
                                   self.guess_resonance_params,
                                   ['B0', 'Delta', 'S', 'A', 'offset'],
                                   self.BFIELD_DIM, self.X_DATA_VAR,
                                   bounds = (lobounds, upbounds), **kwargs)
        
    def fit_param_ang_dep(self, fit_func, guess_func, yname,
                                         param_names,  **kwargs):
        """
        Fits angular dependence of some previously fit parameters. Thin wrapper
        around fit_dataset
        
        Parameters
        ----------
        fit_func : function
            Function to fit
        guess_func : function
            function to generate guesses of parameters. Arguments must be:
            - 1D numpy array of y data
            - numpy array of x data
            - keyword arguments, with the keywords being all dims of ds besides
            xname. Values passed will be individual floats of coordinate values
            corresponding to those dims.
            All arguments must be accepted, not all must be used.
            Must return a list of guesses to the parameters, in the order given in
            param_names
        yname : str
            Name of parameter (data_var) which we will fit over
        param_names : list of str
            Names of parameters of fit_func, in order.
        **kwargs
            can be:
            - names of dims of ds (cannot include xname)
            values should eitherbe single coordinate values or lists of coordinate
            values of those dims. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of curve_fit
            
        Returns
        -------
        None
            Saves resulting analyzedFit object to a new attribute, named 
            (yname)_azimuth_fit
        """
        if self.resonance_fits is None:
            raise AttributeError("Need to run fit_resonances first")

        setattr(self, yname + '_azimuth_fit', fit_dataset(
            self.resonance_fits.fit_ds, fit_func, guess_func, param_names,
                          self.ANGLE_DIM, yname, yname+'_err', **kwargs))