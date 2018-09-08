import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit
from scipy.signal import savgol_filter
import pandas as pd

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis, analyzedFit
from .baseAnalysis import get_coord_selection, fit_dataset
from ..procedures import daedalus_STFMRProcedure

class daedalus_STFMRAnalysis(baseAnalysis):
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
    fit_ds : xarray.Dataset or None
        dataset containing fit parameters. ``None`` if fitting has not been done.
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
        self.series_swept_params = [self.ANGLE_DIM, self.FREQUENCY_DIM]
        self.procedure = daedalus_STFMRProcedure

        self.fit_ds = None
        self.analyzed_fit = None
        self.combined_angle_fit_params = None

    def load_utilsweep(self, direc, fnames = [], header = ''):
        """
        Loads utilsweep. If fnames is not specified, assumes only utilsweep
        files are in the directory and loads them all.

        Parameters
        ----------
        direc : str
            Directory containing UtilSweep files to include in the sweep.
        fnames : list of str, optional
            List of additional data files to include
        header : str, optional
            beginning of the data filenames
        """

        if not fnames:
            for f in os.listdir(direc):
                if f.startswith(header) and os.path.isfile(os.path.join(direc, f)):
                    fnames.append(f)

        if not os.path.isdir(direc):
            raise ImportError("Given directory does not exist!")

        procedure_swept_col = self.BFIELD_DIM
        series_swept_params = [self.FREQUENCY_DIM, self.ANGLE_DIM]

        ex_result = pd.read_table(os.path.join(direc,fnames[0]))

        # record data columns, except one we swept over. Will handle it separately
        file_data_cols = ['LockinOnex']
        data_cols = ['X']
        col_size = ex_result['Field(nominal)'].size
        new_dims = tuple([procedure_swept_col] + series_swept_params)
        swept_col_data = ex_result['Field(nominal)'].values

        # need data_var data to have the correct shape as an array so that
        # all coordinates are taken seriously by the dataset. This exists to
        # reshape the data column into the correct shape
        reshape_helper = [1]*len(series_swept_params)

        # we need to get the ball rolling. Add the first data to growing_ds
        # create data_vars, appropriately reshaped
        new_data_vars = {}
        for col, fcol in zip(data_cols, file_data_cols):
            new_data_vars[col] = (
                new_dims,
                ex_result[fcol].values.reshape(col_size,*reshape_helper)
            )
        # create new columns, with all from series_swept_params only having one
        # coordinate value (hence the need for reshaping)
        phi0 = fnames[0].split('_')[1]
        f0 = fnames[0].split('_')[7]
        params0 = (float(f0), float(phi0))
        new_coords = {procedure_swept_col: swept_col_data}
        for i, param in enumerate(series_swept_params):
            new_coords[param] = np.array([params0[i]])

        growing_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords)

        # load the rest of the procedure data files
        # if ur lookin here b/c you got an error and were trying to load A SINGLE
        # procedure data file 1. i'm sorry this is implemented poorly 2. ur a dumbass
        for f in fnames[1:]:
            rslt = pd.read_table(os.path.join(direc,f))

            # new data vars (but same dims)
            new_data_vars = {}
            for col, fcol in zip(data_cols, file_data_cols):
                new_data_vars[col] = (
                    new_dims,
                    rslt[fcol].values.reshape(col_size,*reshape_helper)
                )
            # new coords (but same dims)
            phi0 = f.split('_')[1]
            f0 = f.split('_')[7]
            params0 = (float(f0), float(phi0))
            new_coords = {procedure_swept_col: swept_col_data}
            for i, param in enumerate(series_swept_params):
                new_coords[param] = [params0[i]]

            # make dataset corresponding to new procedure data file
            fresh_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords)

            growing_ds = fresh_ds.combine_first(growing_ds)

        # sort so all coordinates are in a sensible order
        growing_ds = growing_ds.sortby(list(growing_ds.dims))

        self.sweep_ds = growing_ds # Now fully grown :')
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

    def resonance_model(self, p, B):
        """Convenience function used for fitting with leastsq.
        Unpacks parameters and passes them to total_lor"""
        B0, Delta, S, A, offset = p
        return self.total_lor(B,B0,Delta,S,A,offset)

    def resonance_residual(self, p, B, data, error):
        """
        Function calculating residulals from STFMR resonances, for use with
        leastsq.

        Harsh penalty for Delta parameter being negative or the constant offset
        being much larger than the mean.

        p = [B0, Delta, S, A, offset]
        """

        if p[1]<0:
            penalty = np.empty(data.shape)
            return penalty.fill(1000000000)
        if abs(p[4])>data.mean()+150:
            penalty = np.empty(data.shape)
            return penalty.fill(1000000000)
        return (data - self.resonance_model(p, B))/error

    def fit_resonance(self, da, pguess, error = None):
        """
        Fits a *single* STFMR resonance.

        Parameters
        ----------
        da : xarray.DataArray
            Array to fit over with the field being the only (meaningful)
            dimension and containing lockin X as values.
        pguess : list
            initial guess for the fit parameters. Structure:
            ``p = [B0, Delta, S, A, offset]``
        error : np.array or None
            array of errors to use. If not given, uses 1 for every point.

        Returns
        -------
        popt : list
            List of optimal parameters
        copt : np.array
            The covariance matrix.
        """
        if error is None:
            # TODO: Find a better way of dealing with errors. What we're doing
            # *should* be the same as passing an array of ones, but it is
            # not. We need the constant array to have a reasonable amplitude
            # compared to the data points to not get nonsensical errors.
            filt = savgol_filter(da[self.BFIELD_DIM].values, 5, 3, mode = 'mirror')
            error_amp = np.std(filt - da[self.BFIELD_DIM].values)
            error = error_amp*np.ones(da.values.shape)

        popt, copt, infodict, msg, ier = leastsq(
            self.resonance_residual,
            pguess,
            args = (da[self.BFIELD_DIM].values, da.values,
                    error),
            maxfev = 10000,
            full_output = 1
            )
        return popt, copt

    # TODO: make this guessing better/smarter somehow...
    def guess_resonance_params(self, data, f, angle, Meff, Delta, S45, A45):
        """Guesses parameters for the resonance for this data"""

        offset = data.mean()
        SAsign = np.sign(np.sin(2*(angle - 90)*self.deg2rad)
                         * np.cos((angle - 90)*self.deg2rad)) + 0.01

        return np.array([self.B0(f, Meff), Delta, S45*SAsign, A45*SAsign,
                         offset])

    def fit_resonances(self, Meff_guess, Delta_guess, S45_guess, A45_guess,
                       **selections):
        """
        Fits all of the resonances.

        Fits resonances across the different coordinates scanned in
        the sweep (frequency, angle...) and record the results in a new dataset
        which shares all dims and coordinates except field (since we are fitting
        over that). Separates positive and negative field resonances.

        Parameters
        ----------
        Meff_guess : float
            guess of Meff
        Delta_guess : float
            guess of lorentzian linewidth
        S45_guess : float
            Guess of symmetric amplitude of the lorentzian with ang(B, I)=45
        A45_guess : float
            Guess of antisymmetric amplitude of the lorentzian with ang(B, I)=45
        **selections
            keywords should be names of ``dims`` besides the field dim.
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            have parameter guesses generated. If no selections given, guesses are
            generated for everything

        Returns
        -------
        None
            Saves everything to
            :attr:`~analysis.analysis.daedalus_STFMRAnalysis.daedalus_STFMRAnalysis.fit_ds`
        """
        # fit parameters to save in new dataset
        fit_dvars = ['B0_pos', 'Delta_pos', 'S_pos', 'A_pos', 'offset_pos',
                     'B0_pos_err', 'Delta_pos_err', 'S_pos_err', 'A_pos_err',
                     'offset_pos_err', 'B0_neg', 'Delta_neg', 'S_neg', 'A_neg',
                      'offset_neg', 'B0_neg_err', 'Delta_neg_err', 'S_neg_err',
                      'A_neg_err', 'offset_neg_err']

        # Get coordinate selections and instantiate empty fit dataset
        remaining_dims, coord_combos, self.fit_ds = get_coord_selection(
            self.sweep_ds,
            self.BFIELD_DIM,
            gen_empty_ds = True,
            new_dvar_names = fit_dvars,
            **selections
        )

        # save fit guesses
        self.Meff_guess = Meff_guess
        self.Delta_guess = Delta_guess
        self.S45_guess = S45_guess
        self.A45_guess = A45_guess

        # Fit the resonance for each coord combo and record results
        for combo in coord_combos:
            selection_dict = dict(zip(remaining_dims, combo))
            pos_da = self.sweep_ds[self.X_DATA_VAR].sel(selection_dict).where(
                self.sweep_ds[self.BFIELD_DIM] > 0,
                drop = True
                )
            neg_da = self.sweep_ds[self.X_DATA_VAR].sel(selection_dict).where(
                self.sweep_ds[self.BFIELD_DIM] < 0,
                drop = True
                )

            pos_guess = self.guess_resonance_params(
                pos_da.values,
                selection_dict[self.FREQUENCY_DIM],
                selection_dict[self.ANGLE_DIM],
                self.Meff_guess, self.Delta_guess,
                self.S45_guess, self.A45_guess
            )
            neg_guess = self.guess_resonance_params(
                neg_da.values,
                selection_dict[self.FREQUENCY_DIM],
                selection_dict[self.ANGLE_DIM],
                self.Meff_guess, self.Delta_guess,
                self.S45_guess, self.A45_guess
            )

            # compute the fits and find 1 SD errors on parameters
            # (should be the right way to get errors, based on curve_fit docs)
            # TODO: Handle when fitting fails
            try:
                pos_fit, pos_cov = self.fit_resonance(pos_da, pos_guess)
                neg_fit, neg_cov = self.fit_resonance(neg_da, neg_guess)
                pos_err = np.sqrt(np.diag(pos_cov))
                neg_err = np.sqrt(np.diag(neg_cov))
            except ValueError:
                # if fitting fails, set parameters to nans
                pos_fit, pos_err = pos_guess*np.nan, pos_guess*np.nan
                neg_fit, neg_err = neg_guess*np.nan, neg_guess*np.nan


            # record fit params.
            # IDK a better way to do this assignment unfortunately :(
            self.fit_ds['B0_pos'].loc[selection_dict],  \
            self.fit_ds['Delta_pos'].loc[selection_dict], \
            self.fit_ds['S_pos'].loc[selection_dict], \
            self.fit_ds['A_pos'].loc[selection_dict], \
            self.fit_ds['offset_pos'].loc[selection_dict] = pos_fit
            self.fit_ds['B0_pos_err'].loc[selection_dict],  \
            self.fit_ds['Delta_pos_err'].loc[selection_dict], \
            self.fit_ds['S_pos_err'].loc[selection_dict], \
            self.fit_ds['A_pos_err'].loc[selection_dict], \
            self.fit_ds['offset_pos_err'].loc[selection_dict] = pos_err
            self.fit_ds['B0_neg'].loc[selection_dict], \
            self.fit_ds['Delta_neg'].loc[selection_dict], \
            self.fit_ds['S_neg'].loc[selection_dict], \
            self.fit_ds['A_neg'].loc[selection_dict], \
            self.fit_ds['offset_neg'].loc[selection_dict] = neg_fit
            self.fit_ds['B0_neg_err'].loc[selection_dict], \
            self.fit_ds['Delta_neg_err'].loc[selection_dict], \
            self.fit_ds['S_neg_err'].loc[selection_dict], \
            self.fit_ds['A_neg_err'].loc[selection_dict], \
            self.fit_ds['offset_neg_err'].loc[selection_dict] = neg_err

    def plot_resonances(self, overlay_fits =  False, **selections):
        """
        Plots all resonances subject to the given coordinate constraints.

        Parameters
        ----------
        overlay_fits : bool
            whether to overlay fits on the data
        selections
            keywords should be names of ``dims`` besides the ``field`` dim.
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            have parameter guesses generated. If no selections given, guesses are
            generated for everything

        Returns
        -------
        None
            Plots requested plots.
        """

        if overlay_fits and self.fit_ds is None:
            raise AttributeError('You must fit the resonances before overlaying fits')

        remaining_dims, coord_combos, _ = get_coord_selection(
            self.sweep_ds,
            self.BFIELD_DIM,
            **selections
        )

        # plot the resonances for each coordinate combination
        for combo in coord_combos:
            # construct dict to select appropriate resonance and plot
            selection_dict = dict(zip(remaining_dims, combo))
            self.sweep_ds[self.X_DATA_VAR].sel(selection_dict).plot()
            # overlay fit resonance, if requested
            if overlay_fits:
                # get positive and negative field points
                pos_pts = self.coords[self.BFIELD_DIM].where(
                    self.coords[self.BFIELD_DIM] > 0,
                    drop=True)
                neg_pts = self.coords[self.BFIELD_DIM].where(
                    self.coords[self.BFIELD_DIM] < 0,
                    drop=True)

                # convert this into an evenly spaced, relatively dense domain
                pos_dom = np.linspace(pos_pts.min(), pos_pts.max(), 200)
                neg_dom = np.linspace(neg_pts.min(), neg_pts.max(), 200)

                fit_params = self.fit_ds.sel(selection_dict)

                # Use fit parameters to get the fit function points
                pos_fit = self.total_lor(
                    pos_dom,
                    float(fit_params['B0_pos'].values),
                    float(fit_params['Delta_pos'].values),
                    float(fit_params['S_pos'].values),
                    float(fit_params['A_pos'].values),
                    float(fit_params['offset_pos'].values)
                )
                neg_fit = self.total_lor(
                    neg_dom,
                    float(fit_params['B0_neg'].values),
                    float(fit_params['Delta_neg'].values),
                    float(fit_params['S_neg'].values),
                    float(fit_params['A_neg'].values),
                    float(fit_params['offset_neg'].values)
                )
                plt.plot(pos_dom, pos_fit, color = 'red', zorder=10)
                plt.plot(neg_dom, neg_fit, color = 'red', zorder=10)
            plt.show()

    def plot_fit_resonances(self, **selections):
        """
        Plot just the fits to the resonances.

        Parameters
        ----------
        **selections
            keywords should be names of ``dims`` besides the field dim.
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            have parameter guesses generated. If no selections given, guesses are
            generated for everything

        Returns
        -------
        None
            Plots requested plots.
        """
        if self.fit_ds is None:
            raise AttributeError('You must fit the resonances before overlaying fits')

        remaining_dims, coord_combos, _ = get_coord_selection(
            self.sweep_ds,
            self.BFIELD_DIM,
            **selections
        )

        for combo in coord_combos:
            # construct dict to select appropriate resonance
            selection_dict = dict(zip(remaining_dims, combo))

            # get positive and negative field points
            pos_pts = self.coords[self.BFIELD_DIM].where(
                self.coords[self.BFIELD_DIM] > 0,
                drop=True)
            neg_pts = self.coords[self.BFIELD_DIM].where(
                self.coords[self.BFIELD_DIM] < 0,
                drop=True)

            # convert this into an evenly spaced, relatively dense domain
            pos_dom = np.linspace(pos_pts.min(), pos_pts.max(), 200)
            neg_dom = np.linspace(neg_pts.min(), neg_pts.max(), 200)

            fit_params = self.fit_ds.sel(selection_dict)

            # Use fit parameters to get the fit function points
            pos_fit = self.total_lor(
                pos_dom,
                float(fit_params['B0_pos'].values),
                float(fit_params['Delta_pos'].values),
                float(fit_params['S_pos'].values),
                float(fit_params['A_pos'].values),
                float(fit_params['offset_pos'].values)
            )
            neg_fit = self.total_lor(
                neg_dom,
                float(fit_params['B0_neg'].values),
                float(fit_params['Delta_neg'].values),
                float(fit_params['S_neg'].values),
                float(fit_params['A_neg'].values),
                float(fit_params['offset_neg'].values)
            )
            plt.plot(pos_dom, pos_fit)
            plt.plot(neg_dom, neg_fit)
            plt.xlabel('Field (T)')
            plt.ylabel('Fit resonance')
            plt.title(str(selection_dict)) # ugly but idk how to make it better
            plt.show()

    def separate_field_data(self, phi_offset=270., reverse=True):
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
            Modifies :attr:`~.daedalus_STFMRAnalysis.sweep_ds` in place
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

    def guess_separated_resonance_params(self, X, field, field_azimuth, rf_freq):
        """
        Guesses resonance parameters. For use in
        :meth:`~.daedalus_STFMRAnalysis.fit_separated_resonances`.

        Parameters
        ----------
        X : np.ndarray
            Data to guess parameters of
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
            ``[B0, Delta, S, A, offset]``
        """

        offset = X.mean()
        SAsign = np.sign(np.sin(2*(field_azimuth - 90)*self.deg2rad)
                         * np.cos((field_azimuth - 90)*self.deg2rad)) + 0.01

        return np.array([self.B0(rf_freq, self.Meff_guess),
                         self.Delta_formula(self.alpha_guess, 0, rf_freq),
                         self.S45_guess*SAsign, self.A45_guess*SAsign, offset])

    def fit_separated_resonances(self, Meff_guess, alpha_guess, S45_guess,
                                 A45_guess, **kwargs):
        """
        Fits resonances after
        :meth:`~.daedalus_STFMRAnalysis.separate_field_data`
        is ran.

        This just uses ``curve_fit`` and is a thin wrapper around
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
        kwargs
            can be:
            - names of ``dims`` of
            :attr:`~.daedalus_STFMRAnalysis.sweep_ds`
            values should eitherbe single coordinate values or lists of coordinate
            values of those dims. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of ``curve_fit``

        Returns
        -------
        None
            Saves the results into
            :attr:`~analysis.analysis.daedalus_STFMRAnalysis.daedalus_STFMRAnalysis.res_fit`
            as an :class:`~analysis.analysis.baseAnalysis.analyzedFit` object
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

        self.res_fit = fit_dataset(self.sweep_ds, self.total_lor,
                                   self.guess_separated_resonance_params,
                                   ['B0', 'Delta', 'S', 'A', 'offset'],
                                   self.BFIELD_DIM, self.X_DATA_VAR,
                                   bounds = (lobounds, upbounds), **kwargs)

    def plot_separated_resonances(self, **kwargs):
        """
        Plots just the resonances.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`analysis.analysis.baseAnalysis.plot_dataset`
        """

        plot_dataset(self.sweep_ds, self.BFIELD_DIM, self.X_DATA_VAR, **kwargs)

    def combine_fit_params_inplane(self, phi0 = 270):
        """
        Combines fit parameters if measurements had inplane fields

        Since positive and negative fields, when the field is in-plane,
        correspond to phi and ``phi+180``, we should combine them together into
        a single dataset which goes over a full period in phi. We also negate
        ``phi`` since daedalus rotate's clockwise.

        Parameters
        ----------
        phi0 : float
            the offset angle to shift things by.

        Returns
        -------
        None
            Saves shifted dataset to ``combined_angle_fit_params``
        """

        # make lists of data variable names to extract
        fit_dvars = ['B0', 'Delta', 'S', 'A', 'offset']
        neg_dvars = [dvar + '_neg' for dvar in fit_dvars]
        neg_dvars += [dvar + '_err' for dvar in neg_dvars]
        pos_dvars = [dvar + '_pos' for dvar in fit_dvars]
        pos_dvars += [dvar + '_err' for dvar in pos_dvars]
        fit_dvars += [dvar + '_err' for dvar in fit_dvars]

        # make datasets of positive and negative dvars, rename to just
        # base dvar names
        neg_ds = self.fit_ds.drop(pos_dvars)
        neg_ds.rename(dict(zip(neg_dvars, fit_dvars)), inplace=True)
        pos_ds = self.fit_ds.drop(neg_dvars)
        pos_ds.rename(dict(zip(pos_dvars, fit_dvars)), inplace=True)

        # shift the phi coordinates
        neg_ds.coords[self.ANGLE_DIM] = phi0 - neg_ds.coords[self.ANGLE_DIM].values - 180
        pos_ds.coords[self.ANGLE_DIM] = phi0 - pos_ds.coords[self.ANGLE_DIM].values

        # recombine
        self.combined_angle_fit_params = pos_ds.combine_first(neg_ds)
        # self.combined_analyzed_fit = analyzedFit(
        #     self.combined_angle_fit_params,
        #     self.sweep_ds
        # )

    def fit_azimuthal_dependence_inplane(self, fit_func, guess_func, yname,
                                         param_names,  **kwargs):
        """
        Fits angular dependence of some previously fit parameters. Thin wrapper
        around :func:`~analysis.analysis.baseAnalysis.fit_dataset`

        Parameters
        ----------
        fit_func : function
            Function to fit
        guess_func : function
            function to generate guesses of parameters. Arguments must be:
            - 1D numpy array of y data
            - numpy array of x data
            - keyword arguments, with the keywords being all dims of
            :attr:`~.daedalus_STFMRAnalysis.fit_ds`. Values passed will be
            individual floats of coordinate values corresponding to those ``dims``.
            All arguments must be accepted, not all must be used.
            Must return a list of guesses to the parameters, in the order given in
            param_names
        yname : str
            Name of parameter (``data_var``) which we will fit over
        param_names : list of str
            Names of parameters of ``fit_func``, in order.
        **kwargs
            can be:
            - names of ``dims`` of :attr:`~.daedalus_STFMRAnalysis.fit_ds`
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of ``curve_fit``

        Returns
        -------
        None
            Saves resulting :class:`~analysis.analysis.baseAnalysis.analyzedFit` object
            to a new attribute, named ``(yname)_azimuth_fit``
        """
        if self.combined_angle_fit_params is None:
            raise AttributeError("Need to run combine_fit_params_inplane first")

        setattr(self, yname + '_azimuth_fit', fit_dataset(
            self.combined_angle_fit_params, fit_func, guess_func, param_names,
                          self.ANGLE_DIM, yname, yname+'_err', **kwargs))
