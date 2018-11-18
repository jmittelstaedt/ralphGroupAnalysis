import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis, plot_dataset, fit_dataset
from ..procedures import HallAngProcedure, HallFieldProcedure
from ..procedures import HallCryoAngProcedure, HallCryoFieldProcedure
# from .constants import deg2rad, rad2deg

class HallAnalysis(baseAnalysis):
    """
    Class to contain all second harmonic Hall related functions, and acts as a
    convenient container for importing and storing datasets etc.

    Parameters
    ----------
    scan_type : str
        Should be 'angle' or 'field', representing what was swept within each
        procedure. Defaults to angle

    Attributes
    ----------
    sweep_ds : xarray.Dataset
        Dataset containing the data
    procedure_swept_col : str
        column swept in the procedure
    series_swept_params : list of str
        parameters swept in the series
    procedure : pymeasure.experiment.Procedure
        The procedure class which created the data files. Used for importing
        using PyMeasure
    """

    BFIELD_DIM = 'field_strength'
    ANGLE_DIM = 'field_azimuth'
    TEMP_DIM = 'temperature'
    X2_DATA_VAR = 'X2'
    X1_DATA_VAR = 'X1'

    def __init__(self, scan_type='angle', swept_temp = False):
        """
        Instantiates the analysis object with a nonsensical empty dataset.
        Must load data with a separate method.
        Sets swept procedure column and series swept params, depending on
        what scan_type is. Should be angle for angle sweep procedure, or field
        for field sweep procedure.
        """
        super().__init__()
        if swept_temp:
            if scan_type.lower() == 'angle':
                self.procedure_swept_col = self.ANGLE_DIM
                self.series_swept_params = [self.BFIELD_DIM, self.TEMP_DIM]
                self.procedure = HallCryoAngProcedure
            elif scan_type.lower() == 'field':
                self.procedure_swept_col = self.BFIELD_DIM
                self.series_swept_params = [self.ANGLE_DIM, self.TEMP_DIM]
                self.procedure = HallCryoFieldProcedure
            else:
                raise ValueError("scan_type must be 'field' or 'angle'")
        else:
            if scan_type.lower() == 'angle':
                self.procedure_swept_col = self.ANGLE_DIM
                self.series_swept_params = [self.BFIELD_DIM]
                self.procedure = HallAngProcedure
            elif scan_type.lower() == 'field':
                self.procedure_swept_col = self.BFIELD_DIM
                self.series_swept_params = [self.ANGLE_DIM]
                self.procedure = HallFieldProcedure
            else:
                raise ValueError("scan_type must be 'field' or 'angle'")

    def plot_2harm_angle_dependence(self, **kwargs):
        """
        Plots the second harmonic voltage as a function of field angle.
        Is a thin wrapper around :func:`~analysis.analysis.baseAnalysis.plot_dataset`.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~analysis.analysis.baseAnalysis.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots
        """

        plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                     self.X2_DATA_VAR, **kwargs)

    def plot_2harm_field_dependence(self, **kwargs):
        """
        Plots the second harmonic voltage as a function of field strength. Is a
        thin wrapper around :func:`~analysis.analysis.baseAnalysis.plot_dataset`.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~analysis.analysis.baseAnalysis.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots
        """

        plot_dataset(self.sweep_ds, self.BFIELD_DIM,
                          self.X2_DATA_VAR, **kwargs)

    def plot_1harm_field_dependence(self, **kwargs):
        """
        Plots the first harmonic voltage as a function of field strength. Is a
        thin wrapper around :func:`~analysis.analysis.baseAnalysis.plot_dataset`.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~analysis.analysis.baseAnalysis.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots
        """

        plot_dataset(self.sweep_ds, self.BFIELD_DIM,
                          self.X1_DATA_VAR, **kwargs)

    def plot_1harm_angle_dependence(self, **kwargs):
        """
        Plots the first harmonic voltage as a function of field angle. Is a
        thin wrapper around :func:`~analysis.analysis.baseAnalysis.plot_dataset`.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~analysis.analysis.baseAnalysis.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots
        """

        plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                     self.X1_DATA_VAR, **kwargs)

    @staticmethod
    def first_harmonic_model(phi, IRp, phi0, offset):
        """
        Expected first harmonic signal for in-plane magnetized samples.

        Parameters
        ----------
        phi : float
            Azimuthal angle of the magnetization
        IRp : float
            Planar Hall resistance coefficient times current amplitude
        phi0 : float
            Offset angle
        offset : float
            constant offset
        """
        return 0.5*IRp*np.sin(2*deg2rad*(phi-phi0)) + offset

    @staticmethod
    def first_harmonic_guess(X1, phi, **kwargs):
        """
        Function for generating guesses for
        :meth:`~.HallAnalysis.first_harmonic_model`

        Returns
        -------
        list of float
            parameter guesses in the order [IRp, phi0, offset]
        """
        return [2*np.max(np.abs(X1-np.mean(X1))), 0, np.mean(X1)]

    def fit_first_harmonic(self, **kwargs):
        param_names = ['IRp', 'phi0', 'offset']

        # restrict phi0 to be in [-180,180]
        lobounds = [-np.inf, -180., -np.inf]
        upbounds = [np.inf, 180., np.inf]

        return fit_dataset(self.sweep_ds, HallAnalysis.first_harmonic_model,
                    HallAnalysis.first_harmonic_guess, param_names,
                    self.ANGLE_DIM, self.X1_DATA_VAR,
                    bounds=(lobounds, upbounds), **kwargs)

    @staticmethod
    def FL_y_signal_inplane(phi, H, HFLy, phi0, IRp, offset):
        """
        Expected signal from a y field-like torque. Probably dominant at low
        fields
        """
        return IRp*np.cos(2*deg2rad*(phi-phi0))*np.cos(deg2rad*(phi-phi0))*HFLy/H + offset

    @staticmethod
    def FL_x_signal_inplane(phi, H, HFLx, phi0, IRp, offset):
        """
        Expected signal from a x field-like torque. Probably dominant at low
        fields.
        """
        return IRp*np.cos(2*deg2rad*(phi-phi0))*np.sin(deg2rad*(phi-phi0))*HFLx/H + offset

    @staticmethod
    def AD_y_signal_inplane(phi, H, Hk, HADy, phi0, IRa, offset):
        """
        Expected signal from a y antidamping torque. Probably dominant at high
        fields.
        """
        return IRa*0.5*HADy*np.cos(deg2rad*(phi-phi0))/(H + Hk) + offset

    @staticmethod
    def AD_x_signal_inplane(phi, H, Hk, HADx, phi0, IRa, offset):
        """
        Expected signal from a x antidamping torque. Probably dominant at high
        fields.
        """
        return IRa*0.5*HADx*np.sin(deg2rad*(phi-phi0))/(H + Hk) + offset

    # TODO: figure out what kind of fitting is needed
