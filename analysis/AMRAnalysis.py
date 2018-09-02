import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import leastsq, curve_fit

from pymeasure.experiment import Results
from .baseAnalysis import baseAnalysis, parse_series_file, plot_dataset
from .baseAnalysis import get_coord_selection, fit_dataset


# TODO: use new import/fitting/plotting functions.
# TODO: pretty up docstrings

class AMRAnalysis(baseAnalysis):

    BFIELD_DIM = 'field_strength'
    ANGLE_DIM = 'field_azimuth'
    X_DATA_VAR = 'X'
    WHEATSTONE_R1 = 'wheatsone_R1'
    WHEATSTONE_R2 = 'wheatsone_R2'
    WHEATSTONE_R3 = 'wheatsone_R3'
    WHEATSTONE_VOLTAGE = 'applied_voltage'

    rad2deg = 180.0/np.pi
    deg2rad = np.pi/180.0

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

    def load_sweep(self, direc, series_filename = None, procedure_files = []):
        """
        Is a wrapper around general load sweep so that we can still extract
        wheatstone resistances and add them as attributes
        """

        # TODO: check if this actually works
        super().load_sweep(direc, series_filename, procedure_files)

        # use one procedure to extract Wheatstone voltage and resistances
        if series_filename is not None:
            all_procedure_files, _, _ = parse_series_file(direc, series_filename)
        else:
            all_procedure_files = procedure_files
        ex_result = Results.load(os.path.join(direc,all_procedure_files[0]))
        self.sweep_ds.attrs = {
            "R1": getattr(ex_result.procedure,self.WHEATSTONE_R1),
            "R2": getattr(ex_result.procedure,self.WHEATSTONE_R2),
            "R3": getattr(ex_result.procedure,self.WHEATSTONE_R3),
            "Vs": getattr(ex_result.procedure,self.WHEATSTONE_VOLTAGE)
        }

    def wheatstone_Rx(self, Vm, R1, R2, R3, Vs):
        """
        Computes the resistance in a wheatsone bridge.

        Vm: measured voltage of the magnet/sample
        Vs: voltage driving current in circuit
        R3: Fixed resisor in-line with Rx
        R2: Variable resistor, across from Rx
        R1: Fixed resistor in-line with R2

        Formula was derived by hand and checked with Wikipedia (also we use
        Wikipedia's naming convention)
        """
        A = R2/(R1+R2) - Vm/Vs
        return A*R3/(1-A)

    def calculate_AMR_resistance(self):
        """
        Calculates the resistance of the sample given the voltage across
        the wheatstone bridge. Saves it as a new data_var in the sweep_ds
        """

        if 'resistance' in self.data_vars:
            raise AttributeError('Resistance data already exists.')
        else:
            # calculate the resistances, returns a DataArray
            R = self.wheatstone_Rx(
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
            
    def plot_AMR_angle_dependence(self, field_sel = None):
        """ Plots AMR angle dependence, either at a specified field
        or for all fields """

        if 'resistance' not in self.data_vars:
            raise AttributeError(
                'resistance data must be computed with calculate_AMR_resistance.')

        if field_sel is not None:
            selection = {self.BFIELD_DIM: field_sel}
            plot_dataset(self.sweep_ds, self.ANGLE_DIM, 'resistance',
                              **selection)
        else:
            plot_dataset(self.sweep_ds, self.ANGLE_DIM, 'resistance')

    def plot_AMR_field_dependence(self, angle_sel = None):
        """ Plots AMR field dependence, either at a specified angle
        or for all angles """

        if 'resistance' not in self.data_vars:
            raise AttributeError(
                'resistance data must be computed with calculate_AMR_resistance.')

        if angle_sel is not None:
            selection = {self.ANGLE_DIM: angle_sel}
            plot_dataset(self.sweep_ds, self.BFIELD_DIM, 'resistance',
                              **selection)
        else:
            plot_dataset(self.sweep_ds, self.BFIELD_DIM, 'resistance')

    def fit_AMR_polycrystalline(self):
        """
        Fits the angular dependence
        """
        param_names = ['phi0', 'Ramr', 'phiE', 'an_ratio', 'R0amr']
        # TODO: finish

    def model_angle_uniaxial(self, phi, phi0, Ramr, phiE, an_ratio, R0amr):
        """
        AMR Angle dependence assuming there is an easy axis.
        Did some reshuffling using cos^2(x) = 1/2 + 1/2cos(2x). It's what
        Greg did so probably cos(2X) is easier to fit to?
        Formula is then (R0 + Ramr/2) + Ramr/2cos(2phiM)
        """
        phiM = phi - phi0 - 0.5*an_ratio*np.sin(2*self.deg2rad
                                                             *(phi-phi0-phiE))
        return R0amr + 0.5*Ramr*np.cos(2*self.deg2rad*phiM)

    def guess_uniaxial(self, resistance, phi, field_strength):
        """
        Generates guess parameters for modelling AMR with uniaxial anisotropy.
        For use with make_fit_dataset_guesses.
        param_names = [phi0, Ramr, phiE, an_ratio, R0amr]
        """
        # TODO: create internal variables for guesses on phi0, phiE and an_ratio
        # to be passed in the wrapper guessing function
        return [0.001, self.sweep_ds.attrs['R2']/1000., 20., 0.00001,
                self.sweep_ds.attrs['R2']]
        
    def fit_AMR_uniaxial(self):
        """
        Fits the AMR to a uniaxial anisotropy model
        """
        param_names = ['phi0', 'Ramr', 'phiE', 'an_ratio', 'R0amr']
        setattr(self, 'AMR_azimuth_fit', fit_dataset(
            self.sweep_ds, self.model_angle_uniaxial,
                self.guess_uniaxial, param_names, self.ANGLE_DIM, 'resistance'))

    def model_angle_polycrystalline(self, phi, phi0, Ramr, R0amr):
        """
        Standard AMR angular dependence for polycrystalline/amorphous
        materials. For use in fitting.
        Did some reshuffling using cos^2(x) = 1/2 + 1/2cos(2x). It's what
        Greg did so probably cos(2X) is easier to fit to?
        Formula is then (R0 + Ramr/2) + Ramr/2cos(2phi)
        """
        return R0amr + 0.5*Ramr*np.cos(2*self.deg2rad*(phi-phi0))

    def guess_polycrystaline(self, resistance, phi, field_azimuth):
        """
        Generates guess parameters for modelling AMR with no crysal anisotropy.
        For use with make_fit_dataset_guesses.
        param_names = [phi0, Ramr, R0amr]
        """
        return [0., resistance.max() - self.sweep_ds.attrs['R2'],
                self.sweep_ds.attrs['R2']]
