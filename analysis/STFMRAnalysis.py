import os
from itertools import product
from inspect import getfullargspec

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import pandas as pd

from .baseAnalysis import baseAnalysis, parse_series_file, load_procedure_files
from .dataset_manipulation import get_coord_selection, fit_dataset, plot_dataset
from .dataset_manipulation import analyzedFit
from .converters import STFMRConverter
from .constants import *

class STFMRAnalysis(baseAnalysis):
    """
    Class to contain all STFMR related functions, and acts as a convenient
    container for importing and storing datasets etc.

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
    """

    # PyMeasure procedure names of parameters, for accessing things from
    # xarray Datasets
    BFIELD_DIM = 'field_strength'
    ANGLE_DIM = 'field_azimuth'
    FREQUENCY_DIM = 'rf_freq'
    TEMPERATURE_DIM = 'temperature'
    BIAS_DIM = 'dc_bias'
    X_DATA_VAR = 'X'

    def __init__(self, swept_temp = False, swept_bias = False):
        """
        Instantiates the analysis object with an empty dataset with the correct
        dimension names. Records the guesses for initial parameters for
        fitting of the resonances.

        Must load data with a separate method.
        """

        super().__init__()
        self.procedure_swept_col = self.BFIELD_DIM
        self.series_swept_params = [self.ANGLE_DIM, self.FREQUENCY_DIM]

        if swept_temp:
            self.series_swept_params.append(self.TEMPERATURE_DIM)
        if swept_bias:
            self.series_swept_params.append(self.BIAS_DIM)

        self.codename_converter = STFMRConverter

    # TODO: add a remove_offset function to get rid of offsets in X voltage?

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

        # ensure all angles are in [0,360]
        nfield.coords[self.ANGLE_DIM].values = [x+360 if x<0 else x for x in nfield.coords[self.ANGLE_DIM].values]
        pfield.coords[self.ANGLE_DIM].values = [x+360 if x<0 else x for x in pfield.coords[self.ANGLE_DIM].values]

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

    @staticmethod
    def total_lor_jac(B, B0, Delta, S, A, offset):
        """
        Computes the jacobian of the total lorenzian function to make fitting
        more robust

        Some words about what function this is the jacobian of:
        inside curve_fit, both the function you're fittng and this jacobian 
        will be recast into functions which accept the fit parameters and 
        output a vector in the space of residuals between your function
        evaluated at the input `x` values and the `y` values you supplied to fit
        to. So, `f`: params -> residuals, and this is the function we're
        evaluating the jacobian of. Therefore, we need to output a matrix
        of shape (# values in fit domain) x (# fit parameters) where row `i` is
        the derivatives of f wrt the fit parameters evaluated at `x[i]` and col
        `j` is the derivative of `f` wrt fit parameter `j` evaluated at each `x`
        """
        
        # Define some often-used expressions
        dB = B-B0
        lorentz_denom = (dB**2 + Delta**2)

        # calculate factors which appear in several derivatives
        BD_deriv_factor = (2*S*Delta*dB + A*(dB**2-Delta**2))/lorentz_denom**2
        SA_deriv_factor = Delta/lorentz_denom

        # calculate the derivatives wrt the different fit parameters
        B0_deriv = Delta*BD_deriv_factor
        Delta_deriv = dB*BD_deriv_factor
        S_deriv = Delta*SA_deriv_factor
        A_deriv = dB*SA_deriv_factor
        offset_deriv = np.ones((B.size, 1))

        # Reshape and put together the derivatives into the jacobian
        return np.hstack((B0_deriv[:,np.newaxis],
                          Delta_deriv[:,np.newaxis],
                          S_deriv[:,np.newaxis],
                          A_deriv[:,np.newaxis],
                          offset_deriv))

    def plot_resonances(self, **kwargs):
        """
        Plots just the resonances.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~dataset_manipulation.plot_dataset`
        """

        plot_dataset(self.sweep_ds, self.BFIELD_DIM, self.X_DATA_VAR, **kwargs)

    # TODO: make static method. Need to figure out how to deal with the
    # guesses for Meff, S/A45 and alpha.
    def guess_resonance_params(self, field, X, field_azimuth, rf_freq,
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
        :func:`~dataset_manipulation.fit_dataset`.

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
            values of those ``dims``. Only data withconstants coordinates given by selections
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
                           bounds = (lobounds, upbounds),
                           jac=STFMRAnalysis.total_lor_jac, **kwargs)

        return resonanceFit.from_analyzedFit(afit)

class resonanceFit(analyzedFit):
    """
    Class to contain methods related to STFMR resonance fit parameters. Wrapper
    around :class:`~baseAnalysis.analyzedFit`
    """

    ANGLE_DIM = 'field_azimuth'
    FREQUENCY_DIM = 'rf_freq'
    TEMPERATURE_DIM = 'temperature'
    BIAS_DIM = 'dc_bias'

    @staticmethod
    def from_analyzedFit(afit):
        """
        Returns an instance of this class starting from an analyzedFit instance

        Parameters
        ----------
        afit : analysis.baseAnalysis.analyzedFit instance
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
    def B0_guess(f, B, **kwargs):
        """
        Guess function for use with fitting to find Meff. kwargs is used
        to catch any possible unexpected fit_ds dims, since these should not
        play a role in the fitting
        """
        return [1]

    @staticmethod
    def Delta_formula(f, alpha, offset):
        """Expected formula for the Delta parameter in lorentzians"""
        return alpha*2*np.pi*f/gamma+offset
    
    @staticmethod
    def Delta_formula_guess(f, Delta, **kwargs):
        """ Guess function for use with fitting linewidth to find damping.
        kwargs is used to catch any possible unexpected fit_ds dims. """

        m_guess = (Delta.max() - Delta.min())/(f.max() - f.min())
        Delta0_guess = Delta[0] - m_guess*f[0]

        alpha_guess = m_guess*gamma/(2*np.pi)

        return [alpha_guess, Delta0_guess]

    def fit_ang_dep(self, fit_func, guess_func, yname, param_names, **kwargs):
        """
        Fits angular dependence of some previously fit parameters. Thin wrapper
        around :func:`~dataset_manipulation.fit_dataset`.

        Parameters
        ----------
        fit_func : function
            function to fit data to
        guess_func : function
            function to generate guesses of parameters. Arguments must be:
            - numpy array of x data
            - 1D numpy array of y data
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

    def find_damping(self, **kwargs):
        """
        Fits the linewidth as a function of frequency to find the gilbert 
        Damping parameter.

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
            An object showing the result of the fitting. Contains the damping
            as a function of any other dimensions.
        """

        return self.fit_params(resonanceFit.Delta_formula, resonanceFit.Delta_formula_guess,
                               ['alpha', 'Delta0'], self.FREQUENCY_DIM, 'Delta', **kwargs)

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
        # TODO: Should add Ms as a parameter to be passed and change the Meff
        # outside of the parentheses to it as it is supposed to be
        SHE = self.fit_ds['S']/self.fit_ds['A']*echarge*Meff \
               *tmag*tnm/hbar*np.sqrt(1 + Meff/self.fit_ds['B0'])
        SHE = SHE.rename('SHE')
        return SHE

    def fit_bias_linewidth_change(self, **kwargs):
        """
        Fits the linewidth change as a function of DC bias.

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
            An object showing the result of the fitting. Contains 
            d(Delta)/d(I_DC) as a function of any other dimensions.
        """
        def linfunc(D, m, b):
            return m*D + b
        
        def linfunc_guess(Idc, D, **guess_kwargs):
            m_guess = (D.max() - D.min())/(Idc.max() - Idc.min())

            return [m_guess, 0]

        return biasLinewidthFit.from_analyzedFit(
            self.fit_params(linfunc, linfunc_guess, ['dDdI', 'b'], self.BIAS_DIM,
            "Delta", **kwargs)
            )

    def plot_resonance_components(self, res_parts=("S","A","total"), 
                                  overlay_data=True, hide_large_errors=True,
                                  pts_per_plot=200, **kwargs):
        """ 
        Plots the resonance data, the total lorentzian, and the symmetric
        and antisymmetric parts on the same plot. Basically an extension of
        :meth:`~baseAnalysis.analyzedFit.plot_fits`

        Parameters
        ----------
        res_parts : list of str
            Determines which parts of the resonance to plot. Possibilities
            include `"S"` for the symmetric part, `"A"` for the anti-symmetric part,
            `"total"` for the total lorentzian and `"offset"` for the offset
        overlay_data : bool
            Whether to overlay the resonance parts on the raw data
        hide_large_errors : bool
            Whether to hide very large error bars
        pts_per_plot : int
            How many points to use for the domain in the resonance parts.
        **kwargs
            Can either be:
            - names of ``dims`` of ``fit_ds``. values should eitherbe single coordinate
            values or lists of coordinate values of those ``dims``. Only data with
            coordinates given by selections are plotted. If no selections given,
            everything is plotted.
            - kwargs passed to ``plot`` or ``errorbar``, as appropriate
        
        Returns
        -------
        None
            Just plots and shows the requested things
        """

        remaining_dims = list(self.dims.keys())
        selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in self.fit_ds.dims}
        coord_combos = product(
            *[np.array(self.fit_ds.sel(selections).coords[dimname].values, ndmin = 1)
            for dimname in remaining_dims])

        # Generate domain
        fit_dom = np.linspace(self.main_ds.coords[self.xname].values.min(),
                              self.main_ds.coords[self.xname].values.max(),
                              pts_per_plot)
        
        # Determine which kwargs can be passed to plot
        if self.yerr_name is None:
            plot_argspec = getfullargspec(Line2D)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
        else:
            ebar_argspec = getfullargspec(plt.errorbar)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}
        
        for combo in coord_combos:
            selection_dict = dict(zip(remaining_dims, combo))
            selected_ds = self.fit_ds.sel(selection_dict)
        
            # extract the fit parameters
            B0 = float(selected_ds["B0"].values)
            Delta = float(selected_ds["Delta"].values)
            S = float(selected_ds["S"].values)
            A = float(selected_ds["A"].values)
            offset = float(selected_ds["offset"].values)
            
            # don't plot if there is no meaningful data
            if np.any(np.isnan([B0, Delta, S, A, offset])):
                continue
            
            # fit the requested parts
            fit_ranges = {}
            if 'S' in res_parts:
                fit_ranges['S']=STFMRAnalysis.sym_lor(fit_dom, B0, Delta, S)+offset
            if 'A' in res_parts:
                fit_ranges['A']=STFMRAnalysis.asym_lor(fit_dom, B0, Delta, A)+offset
            if 'total' in res_parts:
                fit_ranges['total']=STFMRAnalysis.total_lor(fit_dom, B0, Delta, S, A, offset)
            if 'offset' in res_parts:
                fit_ranges['offset']=np.ones(fit_dom.size)*offset
            
            # overlay data if requested
            if overlay_data:
                data_dom = self.main_ds.coords[self.xname].values
                data_range = self.main_ds[self.yname].sel(selection_dict).values
                # plot errorbars if available
                if self.yerr_name is not None:
                    yerr = self.main_ds[self.yerr_name].sel(selection_dict).values
                    num_pts = yerr.size
                    errlims = np.zeros(num_pts).astype(bool)
                    if hide_large_errors: # hide outliers if requested
                        data_avg = np.mean(np.abs(data_range))
                        data_std = np.std(data_range)
                        for i, err in enumerate(yerr):
                            if err > 5*data_std:
                                yerr[i] = data_std*.5 # TODO: Find some better way of marking this
                                errlims[i] = True
                        for i, val in enumerate(data_range):
                            if np.abs(val - data_avg) > 5*data_std:
                                data_range[i] = data_avg
                                yerr[i] = data_std*0.5
                                errlims[i] = True
                    plt.errorbar(data_dom, data_range, yerr, lolims=errlims,
                                 uplims=errlims, **plot_kwargs)
                else:
                    plt.plot(data_dom, data_range, **plot_kwargs)
            
            for part in res_parts:
                plt.plot(fit_dom, fit_ranges[part], label=part)
            plt.xlabel(self.xname)
            plt.ylabel(self.yname)
            plt.legend()
            title_str = ''
            for item in selection_dict.items():
                title_str += '{}: {}, '.format(*item)
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.show()

    # TODO: functions for fitting angular dependence of symmetric and 
    # anti-symmetric components

class biasLinewidthFit(analyzedFit):
    """
    Class for analyzing DC bias linewidth changes. Wrapper
    around :class:`~baseAnalysis.analyzedFit`
    """

    LW_CHANGE_VAR = 'dDdI'
    INTERCEPT_VAR = 'b'
    ANGLE_DIM = 'field_azimuth'
    FREQUENCY_DIM = 'rf_freq'
    TEMPERATURE_DIM = 'temperature'

    @staticmethod
    def from_analyzedFit(afit):
        """
        Returns an instance of this class starting from an analyzedFit instance

        Parameters
        ----------
        afit : analysis.baseAnalysis.analyzedFit instance
        """
        return biasLinewidthFit(afit.fit_ds, afit.main_ds, afit.fit_func,
                            afit.guess_func, afit.param_names, afit.xname,
                            afit.yname, afit.yerr_name)

    def fit_sin_dep(self, **kwargs):
        """
        Fits the linewidth/current slope to a sinusoid
        """
        def sinfunc(p, A, phi0):
            phi = p-phi0
            return A*np.sin(phi*deg2rad)
        def sinfunc_guess(p, dDdI, **fit_kwargs):
            Aguess = np.mean(np.abs(dDdI))
            return [Aguess, 0]

        # Making the sign of the amplitude matter by constraining phase
        lobounds = [-np.inf,-90]
        upbounds = [np.inf, 90]

        return self.fit_params(sinfunc, sinfunc_guess, ['Amp','phi0'], 
                               self.ANGLE_DIM, self.LW_CHANGE_VAR,
                               bounds=(lobounds, upbounds), **kwargs)

    def calculate_bias_SHE(self, Meff, B0, Ms, tmag, Anm, x):
        """
        Calculates spin Hall effect from linewidth broadening as a function of
        DC bias current
        From Luqiao's 2011 STFMR paper and adapted a bit
        any parameters can be DataArrays and this should still work.

        Parameters
        ----------
        Meff : float
            Effective magnetization in T
        B0 : float
            Resonant field in T
        Ms : float
            Saturation magnetization in T
        tmag : float
            thickness of the magnetic layer in m
        Anm : float
            cross-sectional area of active layer in m^2
        x : float
            fraction of current flowing through the active layers

        Returns
        -------
        float
            The efficiency of the in-plane polarized torques
        """

        # TODO: add possibility of selecting from dims?

        return self.fit_ds[self.LW_CHANGE_VAR]*gamma*(B0/mu0 + 1/2*Meff/mu0)*Ms*tmag*Anm*2*echarge/(x*2*np.pi*self.fit_ds[self.FREQUENCY_DIM]*hbar)

class STFMRAnalysis2Res(STFMRAnalysis):
    """
    Class to contain all STFMR related functions on data with two resonances,
    and acts as a convenient container for importing and storing datasets etc.

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
    """

    @staticmethod
    def total_2lor(B, B0, Delta1, S1, A1, dB0, Delta2, S2, A2, offset):
        """ Sum of two lorentzians. Defined with one wrt the other so that their
        relative orientation will be consistent (i.e. res field 1 < res field 2) """
        return STFMRAnalysis.total_lor(B, B0, Delta1, S1, A1, offset) + STFMRAnalysis.total_lor(B, B0+dB0, Delta2, S2, A2, 0)

    @staticmethod
    def total_2lor_jac(B, B0, Delta1, S1, A1, dB0, Delta2, S2, A2, offset):
        """
        Computes the jacobian of the sum of two lorenzian functions to make 
        fitting more robust

        Some words about what function this is the jacobian of:
        inside curve_fit, both the function you're fittng and this jacobian 
        will be recast into functions which accept the fit parameters and 
        output a vector in the space of residuals between your function
        evaluated at the input `x` values and the `y` values you supplied to fit
        to. So, `f`: params -> residuals, and this is the function we're
        evaluating the jacobian of. Therefore, we need to output a matrix
        of shape (# values in fit domain) x (# fit parameters) where row `i` is
        the derivatives of f wrt the fit parameters evaluated at `x[i]` and col
        `j` is the derivative of `f` wrt fit parameter `j` evaluated at each `x`
        """
        
        # Define some often-used expressions
        dB1 = B-B0
        lorentz1_denom = (dB1**2 + Delta1**2)

        dB2 = B-B0-dB0
        lorentz2_denom = (dB2**2 + Delta2**2)

        # calculate factors which appear in several derivatives
        BD_deriv1_factor = (2*S1*Delta1*dB1 + A1*(dB1**2-Delta1**2))/lorentz1_denom**2
        SA_deriv1_factor = Delta1/lorentz1_denom

        BD_deriv2_factor = (2*S2*Delta2*dB2 + A2*(dB2**2-Delta2**2))/lorentz2_denom**2
        SA_deriv2_factor = Delta2/lorentz2_denom


        # calculate the derivatives wrt the different fit parameters
        B0_deriv = Delta1*BD_deriv1_factor + Delta2*BD_deriv2_factor
        Delta1_deriv = dB1*BD_deriv1_factor
        S1_deriv = Delta1*SA_deriv1_factor
        A1_deriv = dB1*SA_deriv1_factor

        dB0_deriv = Delta2*BD_deriv2_factor
        Delta2_deriv = dB2*BD_deriv2_factor
        S2_deriv =  Delta2*SA_deriv2_factor
        A2_deriv = dB2*SA_deriv2_factor

        offset_deriv = np.ones((B.size, 1))

        # Reshape and put together the derivatives into the jacobian
        return np.hstack((B0_deriv[:,np.newaxis],
                          Delta1_deriv[:,np.newaxis],
                          S1_deriv[:,np.newaxis],
                          A1_deriv[:,np.newaxis],
                          dB0_deriv[:,np.newaxis],
                          Delta2_deriv[:,np.newaxis],
                          S2_deriv[:,np.newaxis],
                          A2_deriv[:,np.newaxis],
                          offset_deriv))

    def guess_2lor_rel(self, B, X, field_azimuth, rf_freq,
                               **kwargs):
        """
        Guesses resonance parameters for two lorentzians. For use in
        :meth:`~.STFMRAnalysis2Res.fit_resonances`.

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

        Delta_guess = STFMRAnalysis.Delta_formula(self.alpha_guess, 0, rf_freq)
        S_guess = self.S45_guess*SAsign
        A_guess = self.A45_guess*SAsign

        return np.array([STFMRAnalysis.B0(rf_freq, self.Meff_guess),
                         Delta_guess, S_guess, A_guess, self.dB0_guess, 
                         Delta_guess, S_guess, A_guess, offset])

    def fit_resonances(self, dB0_guess, Meff_guess, alpha_guess, S45_guess,
                                 A45_guess, **kwargs):
        """
        Fits resonances after
        :meth:`~.STFMRAnalysis.separate_field_data`
        is ran.

        This uses ``curve_fit`` and is a thin wrapper around
        :func:`~dataset_manipulation.fit_dataset`.

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
            values of those ``dims``. Only data withconstants coordinates given by selections
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
        self.dB0_guess = dB0_guess

        # Bound parameters. Ensures linewidths positive and second resonant 
        # field is always lower than the first.
        param_names =     ['B01',   'Delta1', 'S1',    'A1',    'dB0',    'Delta2', 'S2',    'A2',    'offset']
        lobounds =        [-np.inf, 0,        -np.inf, -np.inf, -np.inf,  0,        -np.inf, -np.inf, -np.inf]
        upbounds =        [ np.inf, np.inf,   np.inf,  np.inf,  0,        np.inf,   np.inf,  np.inf,  np.inf]

        afit = fit_dataset(self.sweep_ds, STFMRAnalysis2Res.total_2lor,
                           self.guess_2lor_rel,
                           param_names,
                           self.BFIELD_DIM, self.X_DATA_VAR,
                           bounds = (lobounds, upbounds),
                           jac=STFMRAnalysis2Res.total_2lor_jac, **kwargs)

        # Calculate actual second resonance field
        afit.fit_ds = afit.fit_ds.assign({
            "B02": lambda x: x.B01 + x.dB0,
            "B02_err": lambda x: np.sqrt(x.B01_err**2 + x.dB0_err**2)
            })

        return resonanceFit2Res.from_analyzedFit(afit)

class resonanceFit2Res(resonanceFit):
    """
    Class to contain methods related to STFMR resonance fit parameters for data
    with two resonances. Extends `~.resonanceFit`
    """

    @staticmethod
    def from_analyzedFit(afit):
        """
        Returns an instance of this class starting from an analyzedFit instance

        Parameters
        ----------
        afit : analysis.baseAnalysis.analyzedFit instance
        """
        return resonanceFit2Res(afit.fit_ds, afit.main_ds, afit.fit_func,
                            afit.guess_func, afit.param_names, afit.xname,
                            afit.yname, afit.yerr_name)

    def find_damping(self, res, **kwargs):
        """
        Fits the linewidth as a function of frequency to find the gilbert 
        Damping parameter.

        Parameters
        ----------
        res : 1 or 2
            which resonance to find Meff of
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

        return self.fit_params(resonanceFit.Delta_formula, resonanceFit.Delta_formula_guess,
                                ['alpha', 'Delta0'], self.FREQUENCY_DIM, 'Delta'+str(res), **kwargs)

    def find_Meff(self, res, **kwargs):
        """
        Fits the resonat field as a function of frequency to find the effective
        magnetization assuming Kittel's formula holds.

        Parameters
        ----------
        res : 1 or 2
            which resonance to find Meff of
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
                               self.FREQUENCY_DIM, 'B0'+str(res), **kwargs)

    def fit_bias_linewidth_change(self, res, **kwargs):
        """
        Fits the linewidth change as a function of DC bias.

        Parameters
        ----------
        res : 0 or 1
            Resonance linewith change to fit to
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
            An object showing the result of the fitting. Contains 
            d(Delta)/d(I_DC) as a function of any other dimensions.
        """
        def linfunc(D, m, b):
            return m*D + b
        
        def linfunc_guess(Idc, D, **guess_kwargs):
            m_guess = (D.max() - D.min())/(Idc.max() - Idc.min())

            return [m_guess, 0]

        return biasLinewidthFit.from_analyzedFit(
            self.fit_params(linfunc, linfunc_guess, ['dDdI', 'b'], self.BIAS_DIM,
            "Delta"+str(res), **kwargs)
            )

    def plot_resonance_components(self, res_parts=("S1","A1","S2","A2","total"), 
                                  overlay_data=True, hide_large_errors=True,
                                  pts_per_plot=200, **kwargs):
        """ 
        Plots the resonance data, the total lorentzian, and the symmetric
        and antisymmetric parts on the same plot. Basically an extension of
        :meth:`~baseAnalysis.analyzedFit.plot_fits`

        Parameters
        ----------
        res_parts : list of str
            Determines which parts of the resonance to plot. Possibilities
            include `"S(1,2)"` for the symmetric part, `"A(1,2)"` for the 
            anti-symmetric part, `"total"` for the total lorentzian and 
            `"offset"` for the offset
        overlay_data : bool
            Whether to overlay the resonance parts on the raw data
        hide_large_errors : bool
            Whether to hide very large error bars
        pts_per_plot : int
            How many points to use for the domain in the resonance parts.
        **kwargs
            Can either be:
            - names of ``dims`` of ``fit_ds``. values should eitherbe single coordinate
            values or lists of coordinate values of those ``dims``. Only data with
            coordinates given by selections are plotted. If no selections given,
            everything is plotted.
            - kwargs passed to ``plot`` or ``errorbar``, as appropriate
        
        Returns
        -------
        None
            Just plots and shows the requested things
        """

        remaining_dims = list(self.dims.keys())
        selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in self.fit_ds.dims}
        coord_combos = product(
            *[np.array(self.fit_ds.sel(selections).coords[dimname].values, ndmin = 1)
            for dimname in remaining_dims])

        # Generate domain
        fit_dom = np.linspace(self.main_ds.coords[self.xname].values.min(),
                              self.main_ds.coords[self.xname].values.max(),
                              pts_per_plot)
        
        # Determine which kwargs can be passed to plot
        if self.yerr_name is None:
            plot_argspec = getfullargspec(Line2D)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
        else:
            ebar_argspec = getfullargspec(plt.errorbar)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}
        
        for combo in coord_combos:
            selection_dict = dict(zip(remaining_dims, combo))
            selected_ds = self.fit_ds.sel(selection_dict)
        
            # extract the fit parameters
            B01 = float(selected_ds["B01"].values)
            B02 = float(selected_ds["B02"].values)
            dB0 = float(selected_ds["dB0"].values)
            Delta1 = float(selected_ds["Delta1"].values)
            Delta2 = float(selected_ds["Delta2"].values)
            S1 = float(selected_ds["S1"].values)
            A1 = float(selected_ds["A1"].values)
            S2 = float(selected_ds["S2"].values)
            A2 = float(selected_ds["A2"].values)
            offset = float(selected_ds["offset"].values)
            
            # don't plot if there is no meaningful data
            if np.any(np.isnan([B01, Delta1, S1, A1, offset, B02, Delta2, S2, A2, dB0])):
                continue
            
            # fit the requested parts
            fit_ranges = {}
            if 'S1' in res_parts:
                fit_ranges['S1']=STFMRAnalysis.sym_lor(fit_dom, B01, Delta1, S1)+offset
            if 'A1' in res_parts:
                fit_ranges['A1']=STFMRAnalysis.asym_lor(fit_dom, B01, Delta1, A1)+offset
            if 'S2' in res_parts:
                fit_ranges['S2']=STFMRAnalysis.sym_lor(fit_dom, B02, Delta2, S2)+offset
            if 'A2' in res_parts:
                fit_ranges['A2']=STFMRAnalysis.asym_lor(fit_dom, B02, Delta2, A2)+offset
            if 'total' in res_parts:
                fit_ranges['total']=STFMRAnalysis2Res.total_2lor(fit_dom, B01, Delta1, S1, A1, dB0, Delta2, S2, A2, offset)
            if 'offset' in res_parts:
                fit_ranges['offset']=np.ones(fit_dom.size)*offset
            
            # overlay data if requested
            if overlay_data:
                data_dom = self.main_ds.coords[self.xname].values
                data_range = self.main_ds[self.yname].sel(selection_dict).values
                # plot errorbars if available
                if self.yerr_name is not None:
                    yerr = self.main_ds[self.yerr_name].sel(selection_dict).values
                    num_pts = yerr.size
                    errlims = np.zeros(num_pts).astype(bool)
                    if hide_large_errors: # hide outliers if requested
                        data_avg = np.mean(np.abs(data_range))
                        data_std = np.std(data_range)
                        for i, err in enumerate(yerr):
                            if err > 5*data_std:
                                yerr[i] = data_std*.5 # TODO: Find some better way of marking this
                                errlims[i] = True
                        for i, val in enumerate(data_range):
                            if np.abs(val - data_avg) > 5*data_std:
                                data_range[i] = data_avg
                                yerr[i] = data_std*0.5
                                errlims[i] = True
                    plt.errorbar(data_dom, data_range, yerr, lolims=errlims,
                                 uplims=errlims, **plot_kwargs)
                else:
                    plt.plot(data_dom, data_range, **plot_kwargs)
            
            for part in res_parts:
                plt.plot(fit_dom, fit_ranges[part], label=part)
            plt.xlabel(self.xname)
            plt.ylabel(self.yname)
            plt.legend()
            title_str = ''
            for item in selection_dict.items():
                title_str += '{}: {}, '.format(*item)
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.show()
    
    # TODO: standard functions for fitting symmetric and anti-symmetric components