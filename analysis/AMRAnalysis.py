import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit

from .baseAnalysis import baseAnalysis, parse_series_file, load_procedure_files
from .dataset_manipulation import fit_dataset, plot_dataset
from .converters import AMRConverter
from .parsers import extract_parameters
from .constants import *

class AMRAnalysis(baseAnalysis):
    """
    Class to contain all AMR related functions, and acts as a convenient
    container for importing and storing datasets etc.

    Parameters
    ----------
    scan_type : str
        Should be ``'angle'`` or ``'field'``, representing what was swept within
         each procedure. Defaults to ``'angle'``
    swept_temp : bool
        Whether temperature was swept in this scan and hence should be a
        dimension of :attr:`.AMRAnalysis.sweep_ds`

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
    TEMP_DIM = 'temperature'
    X_DATA_VAR = 'X'
    WHEATSTONE_R1 = 'wheatsone_R1'
    WHEATSTONE_R2 = 'wheatsone_R2'
    WHEATSTONE_R3 = 'wheatsone_R3'
    WHEATSTONE_VOLTAGE = 'applied_voltage'

    def __init__(self, scan_type='angle', swept_temp = False):
        """
        InstantiaAMRCryoAngProceduretes the analysis object with a nonsensical empty dataset.
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
            elif scan_type.lower() == 'field':
                self.procedure_swept_col = self.BFIELD_DIM
                self.series_swept_params = [self.ANGLE_DIM, self.TEMP_DIM]
            else:
                raise ValueError("scan_type must be 'field' or 'angle'")
        else:
            if scan_type.lower() == 'angle':
                self.procedure_swept_col = self.ANGLE_DIM
                self.series_swept_params = [self.BFIELD_DIM]
            elif scan_type.lower() == 'field':
                self.procedure_swept_col = self.BFIELD_DIM
                self.series_swept_params = [self.ANGLE_DIM]
            else:
                raise ValueError("scan_type must be 'field' or 'angle'")
        self.codename_converter = AMRConverter

    def load_sweep(self, direc, series_filename = None, procedure_files = []):
        """
        This is a wrapper around the general
        :meth:`~baseAnalysis.baseAnalysis.load_sweep` function to allow
        for saving wheatstone resistances as attributes of
        :attr:`~.AMRAnalysis.sweep_ds`

        Parameters
        ----------
        direc : str
            The directory the sweep file is in
        series_file : str
            The name of the series file
        procedure_files : list of str
            Any additional procedure files to include.
        """

        super().load_sweep(direc, series_filename, procedure_files)

        # use one procedure to extract Wheatstone voltage and resistances
        if isinstance(series_filename, str):
            all_procedure_files, _, _ = parse_series_file(direc, series_filename)
        elif isinstance(series_filename, list):
            all_procedure_files, _, _ = parse_series_file(direc, series_filename[0])
        else:
            all_procedure_files = procedure_files

        resistances = extract_parameters(os.path.join(direc,all_procedure_files[0]),
                                         [self.WHEATSTONE_R1,
                                          self.WHEATSTONE_R2,
                                          self.WHEATSTONE_R3,
                                          self.WHEATSTONE_VOLTAGE],
                                         self.codename_converter)
        self.sweep_ds.attrs = {
            "R1": resistances[self.WHEATSTONE_R1],
            "R2": resistances[self.WHEATSTONE_R2],
            "R3": resistances[self.WHEATSTONE_R3],
            "Vs": resistances[self.WHEATSTONE_VOLTAGE]
        }

    @staticmethod
    def wheatstone_Rx(Vm, R1, R2, R3, Vs):
        """
        Computes the resistance in a wheatsone bridge.

        Formula was derived by hand and checked with Wikipedia (also we use
        Wikipedia's naming convention)

        Parameters
        ----------
        Vm : numpy.ndarray
            Measured voltage across the wheatsone bridge
        R1 : float
            Wheatstone R1 resistance, Fixed resistor in-line with R2
        R2 : float
            Variable resistor, across from Rx
        R3 : float
            Wheatstone R3 resistance, in-line with Rx
        Vs : float
            The voltage driving current in the circuit

        Returns
        -------
        numpy.ndarray
            Values of the resistance which was to be measured.
        """
        A = R2/(R1+R2) - Vm/Vs
        return A*R3/(1-A)

    def calculate_AMR_resistance(self):
        """
        Calculates the resistance of the sample given the voltage across
        the wheatstone bridge. Saves it as a new ``data_var`` of
        :attr:~.AMRAnalysis.sweep_ds`.
        """

        if 'resistance' in self.data_vars:
            raise AttributeError('Resistance data already exists.')
        else:
            # calculate the resistances, returns a DataArray
            R = AMRAnalysis.wheatstone_Rx(
                self.sweep_ds[self.X_DATA_VAR],
                self.sweep_ds.attrs['R1'],
                self.sweep_ds.attrs['R2'],
                self.sweep_ds.attrs['R3'],
                self.sweep_ds.attrs['Vs']
            )
            R1 = self.sweep_ds.attrs['R1']
            R2 = self.sweep_ds.attrs['R2']
            R3 = self.sweep_ds.attrs['R3']
            Vs = self.sweep_ds.attrs['Vs']
            # Merge with old dataset, renaming so no conflicts happen
            self.sweep_ds = xr.merge([self.sweep_ds, R.rename('resistance')])
            self.data_vars = self.sweep_ds.data_vars
            self.sweep_ds.attrs = {
                "R1": R1,
                "R2": R2,
                "R3": R3,
                "Vs": Vs
            }

    def plot_AMR_angle_dependence(self, **kwargs):
        """
        Plots the calculated AMR as a function of angle. Is a thin wrapper around
        :func:`~dataset_manipulation.plot_dataset`.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~dataset_manipulation.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots

        Raises
        ------
        AttributeError
            If :meth:`~.AMRAnalysis.calculate_AMR_resistance`
            has not been run.
        """

        if 'resistance' not in self.data_vars:
            raise AttributeError(
                'resistance data must be computed with calculate_AMR_resistance.')

        plot_dataset(self.sweep_ds, self.ANGLE_DIM, 'resistance',
                     **kwargs)

    def plot_AMR_field_dependence(self, **kwargs):
        """
        Plots the calculated AMR as a function of field strength. Is a thin
        wrapper around :func:`~dataset_manipulation.plot_dataset`.

        Parameters
        ----------
        **kwargs
            Passed along directly to :func:`~dataset_manipulation.plot_dataset`

        Returns
        -------
        None
            Just creates the requested plots

        Raises
        ------
        AttributeError
            If :meth:`~.AMRAnalysis.calculate_AMR_resistance`
            has not been run.
        """

        if 'resistance' not in self.data_vars:
            raise AttributeError(
                'resistance data must be computed with calculate_AMR_resistance.')

        plot_dataset(self.sweep_ds, self.BFIELD_DIM, 'resistance',
                     **kwargs)

    def fit_AMR_polycrystalline(self, **kwargs):
        """
        Fits the angular dependence
        """
        param_names = ['phi0', 'Ramr', 'phiE', 'an_ratio', 'R0amr']
        # TODO: finish

    @staticmethod
    def model_angle_uniaxial(phi, phi0, Ramr, phiE, an_ratio, R0amr):
        """
        AMR Angle dependence assuming there is an easy axis.
        Did some reshuffling using cos^2(x) = 1/2 + 1/2cos(2x). It's what
        Greg did so probably cos(2X) is easier to fit to?
        Formula is then (R0 + Ramr/2) + Ramr/2cos(2phiM)
        """
        phiM = phi - phi0 - 0.5*an_ratio*rad2deg\
                                        *np.sin(2*deg2rad*(phi-phi0-phiE))
        return R0amr + 0.5*Ramr*np.cos(2*deg2rad*phiM)

    @staticmethod
    def guess_uniaxial(phi, resistance, **kwargs):
        """
        Generates guess parameters for modelling AMR with uniaxial anisotropy.
        For use with make_fit_dataset_guesses.
        param_names = [phi0, Ramr, phiE, an_ratio, R0amr]
        """
        # TODO: create internal variables for guesses on phi0, phiE and an_ratio
        # to be passed in the wrapper guessing function
        return [1, np.max(np.abs(resistance-np.mean(resistance))), 20., 0.1,
                np.mean(resistance)]

    def fit_AMR_uniaxial(self, **kwargs):
        """
        Fits the AMR to a uniaxial anisotropy model

        **kwargs
            can be:
            - names of ``dims`` of ``ds``
            values should eitherbe single coordinate values or lists of coordinate
            values of those ``dims``. Only data with coordinates given by selections
            are fit to . If no selections given, everything is fit to.
            - kwargs of ``curve_fit``
        """
        param_names = ['phi0', 'Ramr', 'phiE', 'an_ratio', 'R0amr']
        return fit_dataset(self.sweep_ds, AMRAnalysis.model_angle_uniaxial,
                AMRAnalysis.guess_uniaxial, param_names, self.ANGLE_DIM, 'resistance',
                **kwargs)

    @staticmethod
    def model_angle_polycrystalline(phi, phi0, Ramr, R0amr):
        """
        Standard AMR angular dependence for polycrystalline/amorphous
        materials. For use in fitting.
        Did some reshuffling using cos^2(x) = 1/2 + 1/2cos(2x). It's what
        Greg did so probably cos(2X) is easier to fit to?
        Formula is then (R0 + Ramr/2) + Ramr/2cos(2phi)
        """
        return R0amr + 0.5*Ramr*np.cos(2*deg2rad*(phi-phi0))

    @staticmethod
    def guess_polycrystaline(phi, resistance, field_azimuth):
        """
        Generates guess parameters for modelling AMR with no crysal anisotropy.
        For use with make_fit_dataset_guesses.
        param_names = [phi0, Ramr, R0amr]
        """
        return [0., np.max(np.abs(resistance-np.mean(resistance))),
                np.mean(resistance)]
