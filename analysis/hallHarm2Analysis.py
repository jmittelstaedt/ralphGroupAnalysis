import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis

# TODO: use new import/fitting/plotting functions.
# TODO: pretty up docstrings

class Hall2HarmAnalysis(baseAnalysis):

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

    def plot_2harm_field_dependence(self, angle_sel = None):
        """ Plots second harmonic field dependence, either at a specified angle
        or for all angles """

        if angle_sel is not None:
            selection = {self.ANGLE_DIM: angle_sel}
            self.plot_dataset(self.sweep_ds, self.BFIELD_DIM,
                              self.X2_DATA_VAR, **selection)
        else:
            self.plot_dataset(self.sweep_ds, self.BFIELD_DIM,
                              self.X2_DATA_VAR)

    def plot_2harm_angle_dependence(self, field_sel = None):
        """ Plots second harmonic angle dependence, either at a specified field
        or for all fields """

        if field_sel is not None:
            selection = {self.BFIELD_DIM: field_sel}
            self.plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                              self.X2_DATA_VAR, **selection)
        else:
            self.plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                              self.X2_DATA_VAR)

    def plot_1harm_field_dependence(self, angle_sel = None):
        """ Plots first harmonic field dependence, either at a specified angle
        or for all angles """

        if angle_sel is not None:
            selection = {self.ANGLE_DIM: angle_sel}
            self.plot_dataset(self.sweep_ds, self.BFIELD_DIM,
                              self.X1_DATA_VAR, **selection)
        else:
            self.plot_dataset(self.sweep_ds, self.BFIELD_DIM,
                              self.X1_DATA_VAR)

    def plot_1harm_angle_dependence(self, field_sel = None):
        """ Plots first harmonic angle dependence, either at specified fields
        or for all fields """

        if field_sel is not None:
            selection = {self.BFIELD_DIM: field_sel}
            self.plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                              self.X1_DATA_VAR, **selection)
        else:
            self.plot_dataset(self.sweep_ds, self.ANGLE_DIM,
                              self.X1_DATA_VAR)

    def signal_FL_x(self, H, Hk, Ha, theta, phi):
        return 0 # TODO: implement this, should it be fully general or only
        # for in-plane magnetized samples?...

    # TODO: figure out what kind of fitting is needed
