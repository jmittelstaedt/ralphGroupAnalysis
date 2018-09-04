import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis, plot_dataset, fit_dataset

class Hall2HarmAnalysis(baseAnalysis):
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
    """

    BFIELD_DIM = 'field_strength'
    ANGLE_DIM = 'field_azimuth'
    X2_DATA_VAR = 'X2'
    X1_DATA_VAR = 'X1'

    def __init__(self, scan_type='angle'):
        """
        Instantiates the analysis object with a nonsensical empty dataset.
        Must load data with a separate method.
        Sets swept procedure column and series swept params, depending on
        what scan_type is. Should be angle for angle sweep procedure, or field
        for field sweep procedure.
        """
        super().__init__()
        if scan_type.lower() == 'angle':
            self.procedure_swept_col = self.ANGLE_DIM
            self.series_swept_params = [self.BFIELD_DIM]
        elif scan_type.lower() == 'field':
            self.procedure_swept_col = self.BFIELD_DIM
            self.series_swept_params = [self.ANGLE_DIM]
        else:
            raise ValueError("scan_type must be 'field' or 'angle'")

    def plot_2harm_angle_dependence(self, **kwargs):
        """
        Plots the second harmonic voltage as a function of field angle.
        Is a thin wrapper around plot_dataset.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`analysis.baseAnalysis.plot_dataset`

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
        thin wrapper around plot_dataset.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`analysis.baseAnalysis.plot_dataset`

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
        thin wrapper around plot_dataset.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`analysis.baseAnalysis.plot_dataset`

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
        thin wrapper around plot_dataset.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`analysis.baseAnalysis.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots
        """

        plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                     self.X1_DATA_VAR, **kwargs)

    def signal_FL_x(self, H, Hk, Ha, theta, phi):
        return 0 # TODO: implement this, should it be fully general or only
        # for in-plane magnetized samples?...

    # TODO: figure out what kind of fitting is needed
