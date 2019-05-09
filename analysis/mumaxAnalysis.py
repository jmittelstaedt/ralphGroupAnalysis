import os
import re
from itertools import product
from inspect import getfullargspec

import numpy as np
from scipy.integrate import simps
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import pandas as pd

from .baseAnalysis import baseAnalysis, parse_series_file, load_procedure_files
from .dataset_manipulation import get_coord_selection, fit_dataset, plot_dataset
from .STFMRAnalysis import STFMRAnalysis
from .constants import *

# TODO: add plotting method which superimposes the RF current on my

class mumaxSTFMRAnalysis(baseAnalysis):
    """
    Class for importing and integrating STFMR simulations from mumax3
    """

    # TODO: can probably modify the load to just be given a directory, and can
    # then look at the log.txt file to extract meta-parameters and look at the
    # table.txt individually

    def load_sweep(self, sim_dir, dt=None):
        """
        Customized load_sweep to properly load and interpolate the data in a
        mumax3 output table. Can only handle one for now.

        Parameters
        ----------
        sim_dir : str
            Output directory of the mumax3 simulation run, probably ends with .out
        dt : float (optional)
            timestep used for resampling the data. If not given, one is found
            from the frequency of saving the simulation data table
        """

        # Get the runtime and the nominal dt from the output file
        with open(os.path.join(sim_dir, 'log.txt'), 'r') as f:
            simlog = f.read()
            runtime_re = re.search(r'runtime\s*:=\s*(?P<runtime>[^\s]+)\s+', simlog)
            runtime = float(runtime_re['runtime'])
            if dt is None:
                dt_re = re.search(r'tableautosave\((?P<dt>[^\s]+)\)\s+', simlog)
                dt = float(dt_re['dt'])

        # load the raw data, takes a while since the file is yuge
        # TODO: do we want to have the columns it looks for be adjustable? probably should
        raw_data = pd.read_csv(os.path.join(sim_dir, 'table.txt'), delimiter='\t',
            usecols=['# t (s)', 'mx ()', 'my ()', 'mz ()','rf_freq (GHz)',
                     'B_nominal (T)', 'field_azimuth (deg)', 'field_polar (deg)'])
        raw_data = raw_data.rename(columns={
            '# t (s)': 't',
            'mx ()': 'mx',
            'my ()': 'my',
            'mz ()': 'mz',
            'rf_freq (GHz)': 'freq',
            'B_nominal (T)': 'Bnom',
            'field_azimuth (deg)': 'phi',
            'field_polar (deg)': 'theta'
        })
        # convert to G to avoid rounding errors
        raw_data.Bnom = np.around(raw_data.Bnom*1e4).astype(int)

        # TODO: new logic for sweeping through data to make dataset

        Bs = sorted(list(set(raw_data.Bnom)))
        fs = sorted(list(set(raw_data.freq)))
        phis = sorted(list(set(raw_data.phi)))
        thetas = sorted(list(set(raw_data.theta)))
        growing_ds = xr.Dataset()

        # define dimensions of the dataset we want
        new_dims = ('t_rel','field_strength','rf_freq','field_azimuth','field_polar')

        # new relative time array
        rel_ts = np.arange(0,runtime, dt)
        if runtime not in rel_ts:
            rel_ts = np.append(rel_ts, runtime)
        num_ts = rel_ts.size

        # TODO: if we use this, we probably want to do it in
        # the order the loops are in in the code. May be able to speed up some
        # by using the extracted loop values and knowing we're going in
        # sequential order to cache the previous index of the end.
        # also if the number of data points per param vals is a deterministic
        # function of the runtime and tableautosave stuff.
        for phi in phis:
            raw_sel_phi = raw_data[raw_data.phi == phi]
            for theta in thetas:
                raw_sel_theta = raw_sel_phi[raw_sel_phi.theta == theta]
                for f in fs:
                    raw_sel_freq = raw_sel_theta[raw_sel_theta.freq == f]
                    for B in Bs:
                        # selectraw_data the part of the dataframe we care about and get parameter data
                        raw_data_selection = raw_sel_freq[raw_sel_freq.Bnom == B]
                        param_values = {'field_strength': B, 'rf_freq': f, 'field_azimuth': phi, 'field_polar': theta}
                        interp_data = {}

                        # generate array of actual times to interpolate
                        interp_ts = np.linspace(raw_data_selection.t.min(), raw_data_selection.t.max(), num_ts)
                        # interpolate all of the data onto the new time array
                        for col in raw_data_selection.columns:
                            if col in ['Bnom', 'freq', 'phi', 'theta']: # don't need these, will become a coordinates
                                continue
                            tck = interpolate.splrep(raw_data_selection.t.values, raw_data_selection[col].values)
                            interp_data[col] = interpolate.splev(interp_ts, tck)

                        # generate the format needed to define the data vars
                        new_data_vars = {}
                        for col, data_vals in interp_data.items():
                            new_data_vars[col] = (
                                new_dims,
                                data_vals.reshape(num_ts, 1, 1, 1, 1) # hard-coded for just B, f and phi and theta for now
                            )

                        # generate the coordinates, using the relative time array from before
                        new_coords = {k: [v] for k, v in param_values.items()}
                        new_coords['t_rel'] = rel_ts

                        # make new dataset and append to growing one
                        fresh_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords)

                        growing_ds = fresh_ds.combine_first(growing_ds)

        self.sweep_ds = growing_ds
        self.dims = self.sweep_ds.dims
        self.data_vars = self.sweep_ds.data_vars

    # TODO: add more of these types of functions as needed. Maybe make offset
    # and lockin stuff more general, accepting different magnetization components?

    def remove_m_offset(self):
        """
        Removes the offset in the magnetization signals to give better behaved simulated
        resonances
        """
        t_half = self.sweep_ds.t_rel.max()/2
        stable_ts = self.sweep_ds.t_rel.where(self.sweep_ds.t_rel>t_half,drop=True)


        self.sweep_ds['mx'] = self.sweep_ds.mx - (self.sweep_ds.mx.sel(t_rel=stable_ts).min(dim='t_rel') + \
                              self.sweep_ds.mx.sel(t_rel=stable_ts).max(dim='t_rel'))/2
        self.sweep_ds['my'] = self.sweep_ds.my - (self.sweep_ds.my.sel(t_rel=stable_ts).min(dim='t_rel') + \
                              self.sweep_ds.my.sel(t_rel=stable_ts).max(dim='t_rel'))/2
        self.sweep_ds['mz'] = self.sweep_ds.mz - (self.sweep_ds.mz.sel(t_rel=stable_ts).min(dim='t_rel') + \
                              self.sweep_ds.mz.sel(t_rel=stable_ts).max(dim='t_rel'))/2


    def calculate_lockin_m(self):
        """
        Calculates the product of my with the Oersted field oscillation and
        saves the result into another data variable in sweep_ds
        """

        self.sweep_ds = self.sweep_ds.assign(lock_mx = lambda x: x.mx*np.sin(2*np.pi*x.rf_freq*1e9*x.t))
        self.sweep_ds = self.sweep_ds.assign(lock_my = lambda x: x.my*np.sin(2*np.pi*x.rf_freq*1e9*x.t))
        self.sweep_ds = self.sweep_ds.assign(lock_mz = lambda x: x.mz*np.sin(2*np.pi*x.rf_freq*1e9*x.t))

    def standard_Vmix(self, Dmx, Dmy, Dmz):
        return Dmy

    def calculate_vmix(self, func):
        """
        Integrates the lockin my signal to get the simulated Vmix. Returns an
        :class:`~STFMRAnalysis.STFMRAnalysis` object
        """

        if 'lock_my' not in self.sweep_ds.data_vars:
            raise AttributeError("Must run calculate_lockin_m first!")

        # a list to store each Vmix for the different frequencies, necessary
        # since we must integrate over different time periods for each of the
        # different frequencies
        vmixs = []
        for f in self.sweep_ds.rf_freq.values:
            # Finding a two signal period window to integrate over for each frequency
            t_half = self.sweep_ds.t_rel.max()/2
            twoperiods = t_half + 2/f/1e9
            ts = self.sweep_ds.t_rel
            t_window = ts.where(np.logical_and(ts > t_half, ts < twoperiods), drop=True).values

            # integrating using Simpson's rule over the time interval found before
            mx = self.sweep_ds.lock_mx.sel(rf_freq=f, t_rel=t_window).reduce(simps, dim='t_rel', even='last')
            my = self.sweep_ds.lock_my.sel(rf_freq=f, t_rel=t_window).reduce(simps, dim='t_rel', even='last')
            mz = self.sweep_ds.lock_mz.sel(rf_freq=f, t_rel=t_window).reduce(simps, dim='t_rel', even='last')

            vmixs.append(func(mx, my, mz))
        # appending results for the different frequencies
        vmix = xr.concat(vmixs, 'rf_freq')

        # construct an STFMRAnalysis object with the mixing voltage signal
        new_analysis = STFMRAnalysis()
        new_analysis.sweep_ds = xr.Dataset(data_vars={'X': vmix})
        new_analysis.sweep_ds.field_strength.values = new_analysis.sweep_ds.field_strength.values/1e4 # back to T

        return new_analysis

    def plot_rf_overlay(self, pts_per_plot = 200, **kwargs):
        """
        Plots my with an overlay of the RF current.

        Parameters
        ----------

        pts_per_plot : int
            How many points to use for the RF current plot.
        **kwargs
            Can either be:
            - names of ``dims`` of ``sweep_ds``. values should eitherbe single
            coordinate values or lists of coordinate values of those ``dims``.
            Only data with coordinates given by selections are plotted. If no
            selections given, everything is plotted.
            - kwargs passed to ``plot``

        Returns
        -------
        None
            Just plots the requested fits.
        """

        remaining_dims = list(self.dims.keys())
        selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in self.sweep_ds.dims}
        coord_combos = product(
            *[np.array(self.sweep_ds.sel(selections).coords[dimname].values, ndmin = 1)
            for dimname in remaining_dims])

        # determine fit domain
        minfreq = self.sweep_ds.rf_freq.values.min()
        t_half = self.sweep_ds.t_rel.max()/2
        twoperiods = t_half + 2/minfreq/1e9
        ts = self.sweep_ds.t_rel
        t_window = ts.where(np.logical_and(ts > t_half, ts < twoperiods), drop=True).values

        # Determine which kwargs can be passed to plot
        plot_argspec = getfullargspec(Line2D)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}

        for combo in coord_combos:
            # make selections, including time window
            selection_dict = dict(zip(remaining_dims, combo))
            selection_dict['t_rel'] = t_window
            selected_ds = self.sweep_ds.sel(selection_dict)

            # get my data
            data_dom = selected_ds.t_rel.values
            data_range = selected_ds.my.values

            # Calculate RF current oscillations
            amp = np.abs(data_range).max()
            offset = data_range.mean()
            rf_dom = np.linspace(selected_ds.t.values.min(), selected_ds.t.values.max(), pts_per_plot)
            rf_range = amp*np.sin(2*np.pi*selection_dict['rf_freq']*1e9*rf_dom)+offset

            # plot
            plt.plot(data_dom, data_range, label='my', **plot_kwargs)
            plt.plot(rf_dom, rf_range, label='current')
            # add labels and make the title reflect the current selection
            plt.xlabel('t_rel')
            plt.ylabel('my')
            title_str = ''
            for item in selection_dict.items():
                if item[0] == 't_rel': # don't want to plot this selection
                    continue
                title_str += '{}: {}, '.format(*item)
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.legend()
            plt.show()

class mumax2layerSTFMRAnalysis(baseAnalysis):
    """
    Class for importing and integrating STFMR simulations with two FM layers
    from mumax3
    """

    def load_sweep(self, sim_dir, dt=None):
        """
        Customized load_sweep to properly load and interpolate the data in a
        mumax3 output table. Can only handle one for now.

        Parameters
        ----------
        sim_dir : str
            Output directory of the mumax3 simulation run, probably ends with .out
        dt : float (optional)
            timestep used for resampling the data. If not given, one is found
            from the frequency of saving the simulation data table
        """

        with open(os.path.join(sim_dir, 'log.txt'), 'r') as f:
            simlog = f.read()
            runtime_re = re.search(r'runtime\s*:=\s*(?P<runtime>[^\s]+)\s+', simlog)
            if dt is None:
                dt_re = re.search(r'tableautosave\((?P<dt>[^\s]+)\)\s+', simlog)
                dt = float(dt_re['dt'])
            runtime = float(runtime_re['runtime'])

        raw_data = pd.read_csv(os.path.join(sim_dir, 'table.txt'), delimiter='\t')
        raw_data = raw_data.rename(columns={
            '# t (s)': 't',
            'my ()': 'mtoty',
            'm.region1y ()': 'm1y',
            'm.region2y ()': 'm2y',
            'rf_freq (GHz)': 'freq',
            'B_nominal (T)': 'Bnom',
            'field_azimuth (deg)': 'phi'
        })
        # convert to G to avoid rounding errors
        raw_data.Bnom = np.around(raw_data.Bnom*1e4).astype(int)

        Bs = sorted(list(set(raw_data.Bnom)))
        fs = sorted(list(set(raw_data.freq)))
        angs = sorted(list(set(raw_data.phi)))
        growing_ds = xr.Dataset()

        # define dimensions of the dataset we want
        new_dims = ('t_rel','field_strength','rf_freq','field_azimuth')

        # new relative time array
        rel_ts = np.arange(0,runtime, dt)
        if runtime not in rel_ts:
            rel_ts = np.append(rel_ts, runtime)
        num_ts = rel_ts.size

        for a in angs:
            for f in fs:
                for B in Bs:
                    # select the part of the dataframe we care about and get parameter data
                    raw_data_selection = raw_data[np.logical_and(np.logical_and(raw_data.Bnom==B, raw_data.freq==f), raw_data.phi==a)]
                    param_values = {'field_strength': B, 'rf_freq': f, 'field_azimuth': a}
                    interp_data = {}

                    # generate array of actual times to interpolate
                    interp_ts = np.linspace(raw_data_selection.t.min(), raw_data_selection.t.max(), num_ts)
                    # interpolate all of the data onto the new time array
                    for col in raw_data_selection.columns:
                        if col in ['Bnom', 'freq', 'phi']: # don't need these, will become a coordinates
                            continue
                        tck = interpolate.splrep(raw_data_selection.t.values, raw_data_selection[col].values)
                        interp_data[col] = interpolate.splev(interp_ts, tck)

                    # generate the format needed to define the data vars
                    new_data_vars = {}
                    for col, data_vals in interp_data.items():
                        new_data_vars[col] = (
                            new_dims,
                            data_vals.reshape(num_ts, 1, 1, 1) # hard-coded for just B, f and phi for now
                        )

                    # generate the coordinates, using the relative time array from before
                    new_coords = {k: [v] for k, v in param_values.items()}
                    new_coords['t_rel'] = rel_ts

                    # make new dataset and append to growing one
                    fresh_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords)

                    growing_ds = fresh_ds.combine_first(growing_ds)

        self.sweep_ds = growing_ds
        self.dims = self.sweep_ds.dims
        self.data_vars = self.sweep_ds.data_vars

    def remove_my_offset(self):
        """
        Removes the offset in the m_y signal to give better behaved simulated
        resonances
        """
        t_half = self.sweep_ds.t_rel.max()/2
        stable_ts = self.sweep_ds.t_rel.where(self.sweep_ds.t_rel>t_half,drop=True)

        # remove offsets from all components
        for my in ['m1y', 'm2y', 'mtoty']:
            self.sweep_ds[my] = self.sweep_ds[my] \
                - (self.sweep_ds[my].sel(t_rel=stable_ts).min(dim='t_rel') \
                + self.sweep_ds[my].sel(t_rel=stable_ts).max(dim='t_rel'))/2

    def calculate_lockin_my(self):
        """
        Calculates the product of my with the Oersted field oscillation and
        saves the result into another data variable in sweep_ds
        """

        for my in ['m1y', 'm2y', 'mtoty']:
            self.sweep_ds = self.sweep_ds.assign(
                {'lock_%s'%my: lambda x: x[my]*np.sin(2*np.pi*x.rf_freq*1e9*x.t)}
            )

    def calculate_vmix(self):
        """
        Integrates the lockin my signal to get the simulated Vmix. Returns an
        :class:`~STFMRAnalysis.STFMRAnalysis` object
        """

        # TODO: Ensure that calculate_lockin_my has been ran
        for my in ['m1y', 'm2y', 'mtoty']:
            if 'lock_%s'%my not in self.sweep_ds.data_vars:
                raise AttributeError("Must run calculate_lockin_my first!")

        # for saving all the analysis objects in
        analysi = []

        for my in ['m1y', 'm2y', 'mtoty']:
            # a list to store each Vmix for the different frequencies, necessary
            # since we must integrate over different time periods for each of the
            # different frequencies
            vmixs = []
            for f in self.sweep_ds.rf_freq.values:
                # Finding a two signal period window to integrate over for each frequency
                t_half = self.sweep_ds.t_rel.max()/2
                twoperiods = t_half + 2/f/1e9
                ts = self.sweep_ds.t_rel
                t_window = ts.where(np.logical_and(ts > t_half, ts < twoperiods), drop=True).values

                # integrating using Simpson's rule over the time interval found before
                vmixs.append(self.sweep_ds['lock_%s'%my].sel(rf_freq=f, t_rel=t_window).reduce(simps, dim='t_rel', even='last'))

            # appending results for the different frequencies
            vmix = xr.concat(vmixs, 'rf_freq')

            # construct an STFMRAnalysis object with the mixing voltage signal
            new_analysis = STFMRAnalysis()
            new_analysis.sweep_ds = xr.Dataset(data_vars={'X': vmix})
            new_analysis.sweep_ds.field_strength.values = new_analysis.sweep_ds.field_strength.values/1e4 # back to T

            analysi.append(new_analysis)

        return analysi

    def plot_rf_overlay(self, my, pts_per_plot = 200, **kwargs):
        """
        Plots my with an overlay of the RF current.

        Parameters
        ----------
        my : str
            Which my we want to plot, `'m1y'`, `'m2y'` or `'mtoty'`.
        pts_per_plot : int
            How many points to use for the RF current plot.
        **kwargs
            Can either be:
            - names of ``dims`` of ``sweep_ds``. values should eitherbe single
            coordinate values or lists of coordinate values of those ``dims``.
            Only data with coordinates given by selections are plotted. If no
            selections given, everything is plotted.
            - kwargs passed to ``plot``

        Returns
        -------
        None
            Just plots the requested fits.
        """

        remaining_dims = list(self.dims.keys())
        selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in self.sweep_ds.dims}
        coord_combos = product(
            *[np.array(self.sweep_ds.sel(selections).coords[dimname].values, ndmin = 1)
            for dimname in remaining_dims if dimname != 't_rel'])

        # determine fit domain
        minfreq = self.sweep_ds.rf_freq.values.min()
        t_half = self.sweep_ds.t_rel.max()/2
        twoperiods = t_half + 2/minfreq/1e9
        ts = self.sweep_ds.t_rel
        t_window = ts.where(np.logical_and(ts > t_half, ts < twoperiods), drop=True).values

        # Determine which kwargs can be passed to plot
        plot_argspec = getfullargspec(Line2D)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}

        for combo in coord_combos:
            # make selections, including time window
            selection_dict = dict(zip(remaining_dims, combo))
            selection_dict['t_rel'] = t_window
            selected_ds = self.sweep_ds.sel(selection_dict)

            # get my data
            data_range = selected_ds[my].values

            # Calculate RF current oscillations
            amp = np.abs(data_range.max() - data_range.min())
            offset = data_range.mean()
            rf_dom = np.linspace(selected_ds.t.values.min(), selected_ds.t.values.max(), pts_per_plot)
            rf_range = amp*np.sin(2*np.pi*selection_dict['rf_freq']*1e9*rf_dom)+offset
            rf_plot_dom = np.linspace(t_window.min(), t_window.max(), pts_per_plot)

            # plot
            plt.plot(t_window, data_range, label=my, **plot_kwargs)
            plt.plot(rf_plot_dom, rf_range, label='current')
            # add labels and make the title reflect the current selection
            plt.xlabel('t_rel')
            plt.ylabel(my)
            title_str = ''
            for item in selection_dict.items():
                if item[0] == 't_rel': # don't want to plot this selection
                    continue
                title_str += '{}: {}, '.format(*item)
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.legend()
            plt.show()
