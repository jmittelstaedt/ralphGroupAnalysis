import os.path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pymeasure.experiment import Results

import .constants
from .baseAnalysis import parse_series_file
from ..procedures import FMRAnalogTrace, FMRDigitalTrace

def normalize(data):
    return (data - data.min())/(
              data.max() - data.min())*2 - 1

def lorentzian_derivative(H, A, Hres, delta_H):
    """ Lorentzian derviative fitting function
    H: Field (Oe)
    A: Amplitude (a.u.)
    Hres: Lorentzian Peak Extrema (Oe)
    delta_H: Full Width at Half Maximum (Oe)
     """
    return -2.*A*(H-Hres)/((delta_H/2)**2.*(1.+(H-Hres)**2./(delta_H/2)**2.)**2.)

def kittel(H, gamma, Meff, Ha=0):
    """ Kittel formula as a function of field, returns frequency in GHz
    H: Applied field (Oe)
    gamma: Gyromagnetic ratio (GHz/Oe)
    Meff: Effective magnetization (emu/cc)
    Ha: Anisotropy field (Oe)
    """
    return (gamma/(2.*np.pi))*np.sqrt((H + Ha + 4*np.pi*Meff)(H + Ha))

def inverted_kittel(f, gamma, Meff, Ha=0):
    """ Kittel formula as a function of frequency, returns field in Oe
    f: Frequency (GHz)
    gamma: Gyromagnetic ratio (GHz/Oe)
    Meff: Effective magnetization (emu/cc)
    Ha: Anisotropy field (Oe)
    """
    return -(4*np.pi*Meff)/2+np.sqrt((4*np.pi*Meff)**2/4+f**2/gamma**2) - Ha

def restricted_inverted_kittel(f, gamma, Meff, Ha=0):
    """ Returns a very large number if parameters are out-of-bounds to restrict
    the least squares fitting from using those numbers
    """
    if Meff <= 0 or Ha < 0:
        return 1e10
    else:
        return inverted_kittel(f, gamma, Meff, Ha)

def kittel_oop(H, gamma, Meff, Ha=0):
    """ Kittel formula (out-of-plane) as a function of field, returns frequency in GHz
    H: Applied field (Oe)
    gamma: Gyromagnetic ratio (GHz/Oe)
    Meff: Effective magnetization (emu/cc)
    Ha: Anisotropy field (Oe)
    """
    return (gamma/(2.*np.pi))*np.sqrt(H + Ha - 4*np.pi*Meff)

def inverted_kittel_oop(f, gamma,Meff,Ha=0):
    """ Kittel formula (out-of-plane) as a function of frequency, returns field in Oe
    f: Frequency (GHz)
    gamma: Gyromagnetic ratio (GHz/Oe)
    Meff: Effective magnetization (emu/cc)
    Ha: Anisotropy field (Oe)
    """
    return -Ha +4*np.pi*Meff + 2*np.pi*f/gamma

def linear_linewidth(f, gamma, alpha, delta_H0):
    """ Linear form of the linewidth as a function of frequency
    f: Frequency (GHz)
    gamma: Gyromagnetic ratio (GHz/Oe)
    alpha: Damping
    delta_H0: Inhomogeneous linewidth (Oe)
    """
    return 2*alpha/(gamma)*f + delta_H0

def restricted_linear_linewidth(f, gamma, alpha, delta_H0):
    """ Returns a very large number if parameters are out-of-bounds to restrict
    the least squares fitting from using those numbers
    """
    if alpha < 0 or delta_H0 < 0:
        return 1e10
    else:
        return linear_linewidth(f, gamma, alpha, delta_H0)

class FMRTraceAnalysis(object):

    def __init__(self, results, x_axis='Field (G)', y_axis='X (V)',
                 baseline=None, fit_rising_edge=False, edge_precent=0.2):
        self.x_axis, self.y_axis = x_axis, y_axis

        if isinstance(results, pd.DataFrame):
            # Load the pandas DataFrame
            self.data = results
            self.frequency = None # Don't assume frequency
        else:
            # Load the Results object, including its frequency information
            if not isinstance(results, Results):
                results = Results.load(results, procedure_class=AnalogTrace)
            self.results = results

            self.frequency = results.procedure.frequency

            self.reset()

        if baseline is None:
            baseline = lambda x, y: np.poly1d(np.polyfit(x, y, 0))(x)
        self.baseline = baseline
        if fit_rising_edge:
            self.rising_edge_fit = True
            self.rising_edge_precent = edge_precent
            self.fit_rising_edge(precent=edge_precent)
        else:
            self.rising_edge_fit = False
            self.fit_resonance()

    @property
    def x(self):
        return self.data[self.x_axis]

    @property
    def y(self):
        """ Returns the Y data after the baseline function is subtraced
        """
        y = normalize(self.data[self.y_axis])
        return y - self.baseline(self.data[self.x_axis], y)

    def fit(self, x):
        """ Returns the fit function evaluated for the given x (or array of x)
        """
        return lorentzian_derivative(x, self.amplitude, self.resonance, self.linewidth)

    def reset(self):
        """ Overwrite the working data with the version from file """
        log.info("Resetting the working data with the file original")
        self.data = self.results.data

    def resonance_guess(self, x, y):
        """ Finds the resonance based on the fact that the signal only crosses
        zero voltage during resonance
        """
        amplitude_guess = np.abs(y.max() - y.min())

        field_at_max = x[y.idxmax()]
        field_at_min = x[y.idxmin()]
        resonance_guess = field_at_max + (field_at_min - field_at_max)/2
        self.peak_to_peak = field_at_min - field_at_max
        linewidth_guess = np.sqrt(3) * self.peak_to_peak

        return (amplitude_guess, resonance_guess, linewidth_guess)

    def fit_resonance(self, x=None, y=None, guess=None, section=None):
        """ Returns fitting parameters of a Lorentzian derivative
        """
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if guess is None:
            guess = self.resonance_guess(x, y)
        if section is None:
            section = [0, len(x)]
        popt, pcov = curve_fit(lorentzian_derivative, x[section[0]:section[1]], y[section[0]:section[1]], p0=guess)
        self.amplitude, self.resonance, self.linewidth = popt
        self.covariance = pcov
        self.amplitude_std, self.resonance_std, self.linewidth_std = np.sqrt(np.diag(pcov))
        # Enforce a positive linewidth
        self.linewidth = np.abs(self.linewidth)

    def rising_edge_section(self, x, y, precent=None):
        if precent is None:
            precent = self.rising_edge_precent
        end = y.idxmax() + int(precent*(y.idxmin() - y.idxmax()))
        return [0, end]

    def fit_rising_edge(self, x=None, y=None, guess=None, precent=0.2):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        section = self.rising_edge_section(x, y, precent)
        self.fit_resonance(x, y, guess, section=section)

    def rising_edge_error(self, precent=None):
        if precent is None:
            precent = self.rising_edge_precent
        section = self.rising_edge_section(self.x ,self.y, precent)
        error = (self.y[section[0]:section[1]] - self.fit(self.x)[section[0]:section[1]])**2.
        return error.sum()

    def normalize(self, minimum=0, maximum=1):
        """ Normalizes the data and fit based on the minimum and maximum values
        provided
        """
        pass

    def plot(self, fmt='o', fit_points=1000):
        plt.plot(self.x, self.y, fmt, ms=3, mec='#aaaaaa', color='#aaaaaa')
        H = np.linspace(self.x.min(), self.x.max(), fit_points)
        plt.plot(H, self.fit(H), 'r', linewidth=2)
        plt.xlabel('Field (Oe)')
        plt.ylabel('dP/dH (a.u.)')
        min_field = self.resonance - 5*self.linewidth
        max_field = self.resonance + 5*self.linewidth
        plt.xlim(min_field, max_field)
        plt.ylim(-1.2, 1.2)
        return plt

class FMRAnalysis(object):

    COLUMNS = ['Filename', 'Frequency (GHz)', 'Resonance (Oe)', 'Resonance Std (Oe)',
                'Linewidth (Oe)', 'Linewidth Std (Oe)', 'Peak-to-Peak (Oe)']

    def __init__(self, sweep_type="analog"):
        self.traces = {}
        self.gyromagnetic_ratio = None
        self.data = pd.DataFrame(columns=self.COLUMNS)
        if sweep_type == "analog":
            self.procedue =FMRAnalogTrace
        elif sweep_type == "digital":
            self.procedure = FMRDigitalTrace
        else:
            raise ValueError("Bad sweep_type! Must be 'analog' or 'digital'")


    def load_sweep(self, direc, series_file = None, procedure_files = [], **kwargs):
        """
        Loads data into internal reperesentation.
        """
        if not procedure_files and series_file is None:
            raise ImportError("Unable to find files to import!")
        if not os.path.isdir(direc):
            raise ImportError("Given directory does not exist!")

        all_procedure_files = []

        # import procedure data files from sweep files
        if isinstance(series_file, str):
            all_procedure_files, _, _ = parse_series_file(direc, series_file)
        elif isinstance(series_file, list):
            for sfile in series_file:
                auto_procedure_files, _, _ = parse_series_file(direc, sfile)
                all_procedure_files += auto_procedure_files
        else: # Assumed none given
            pass

        # combine with any explicitly given procedure data files
        all_procedure_files += procedure_files
        # make all are unique
        all_procedure_files = list(set(all_procedure_files))

        for fname in all_procedure_files:
            self.add_results(Results.load(os.path.join(direc,fname)), procedure=self.procedure)

        self.data.sort_values(by='Frequency (GHz)')

    def reload(self):
        """ Reloads the analysis data from each FMRTraceAnalysis
        """
        self.data = pd.DataFrame(columns=self.COLUMNS)
        for frequency, trace in self.traces.items():
            if trace.rising_edge_fit:
                trace.fit_rising_edge(precent=trace.rising_edge_precent)
            else:
                trace.fit_resonance()
            packet = {
                 'Filename': trace.results.data_filename,
                 'Frequency (GHz)': trace.frequency,
                 'Resonance (Oe)': trace.resonance,
                 'Resonance Std (Oe)': trace.resonance_std,
                 'Linewidth (Oe)': trace.linewidth,
                 'Linewidth Std (Oe)': trace.linewidth_std,
                 'Peak-to-Peak (Oe)': trace.peak_to_peak,
                }
            self.data = self.data.append([packet], ignore_index=True)
        self.sample = trace.results.procedure.sample

    def add(self, trace):
        self.traces[trace.frequency] = trace
        self.reload()

    def add_results(self, results, **kwargs):
        trace = FMRTraceAnalysis(results, **kwargs)
        self.add(trace)

    def __len__(self):
        return len(self.traces)

    def __contains__(self, frequency):
        return frequency in self.traces

    def __getitem__(self, frequency):
        try:
            return self.traces[frequency]
        except KeyError:
            raise KeyError("GHz frequency does not match any traces")

    @staticmethod
    def fit_linewidth_data(f, delta_H, guess=None, sigma=None, gamma=None):
        if gamma is None:
            log.info("Using constant gyromagnetic ratio: %e GHz/Oe")
            gamma = constants.gamma
        if guess is None:
            log.info("Making automatic guess at linewidth frequency dependence "
                     "fitting parameters")
            guess = (0, 0)
        def linear(f, alpha, delta_H0):
            return restricted_linear_linewidth(f, gamma, alpha, delta_H0)
        popt, pcov = curve_fit(linear,
                f, delta_H,
                p0=guess,
                sigma=sigma,
                absolute_sigma=True
        )
        #TODO: Report fit results to log
        return popt, pcov

    def fit_linewidth(self, guess=None, func=None):
        if func is None:
            func = self.fit_linewidth_data
        if self.gyromagnetic_ratio is None:
            log.warning("Fitting is not using the gyromagnetic ratio from "
                        "resonance frequency dependence")

        popt, pcov = func(
            self.data['Frequency (GHz)'],
            self.data['Linewidth (Oe)'],
            gamma=self.gyromagnetic_ratio,
            sigma=self.data['Linewidth Std (Oe)']
        )

        self.damping, self.linewidth_inhomogeneous = popt
        self.linewidth_covariance = pcov
        self.damping_std, self.linewidth_inhomogeneous_std = np.sqrt(np.diag(pcov))

    @staticmethod
    def fit_resonance_data(f, Hres, guess=None, sigma=None):
        if guess is None:
            guess = (constants.gamma, 100, 0)
        popt, pcov = curve_fit(restricted_inverted_kittel,
            f, Hres,
            p0=guess,
            sigma=sigma,
            absolute_sigma=True
        )
        #TODO: Report fit results to log
        return popt, pcov

    @staticmethod
    def fit_resonance_data_gamma_fixed(f, Hres, gamma=None, guess=None, sigma=None):
        if gamma is None:
            gamma = constants.gamma
        if guess is None:
            guess = (100, 0)
        fit_func = lambda f, Meff, Ha: restricted_inverted_kittel(f, gamma, Meff, Ha)
        popt, pcov = curve_fit(fit_func,
            f, Hres,
            p0=guess,
            sigma=sigma,
            absolute_sigma=True
        )
        #TODO: Report fit results to log
        return popt, pcov

    def fit_resonance(self, guess=None, func=None, **kwargs):
        if func is None:
            func = self.fit_resonance_data
        popt, pcov = func(
            self.data['Frequency (GHz)'],
            self.data['Resonance (Oe)'],
            sigma=self.data['Resonance Std (Oe)'],
            **kwargs
        )
        if func is self.fit_resonance_data_gamma_fixed:
            if 'gamma' in kwargs:
                self.gyromagnetic_ratio = kwargs['gamma']
                self.gyromagnetic_ratio_std = 0
            self.effective_magnetization, self.anisotropy_field = popt
            self.effective_magnetization_std, self.anisotropy_field_std = np.sqrt(np.diag(pcov))
        else:
            self.gyromagnetic_ratio, self.effective_magnetization, self.anisotropy_field = popt
            self.gyromagnetic_ratio_std, self.effective_magnetization_std, self.anisotropy_field_std = np.sqrt(np.diag(pcov))
        self.resonance_covariance = pcov

    def summary(self):
        from IPython.display import HTML
        html = """
        <table>
            <tr>
                <td>Sample:</td><td>{sample}</td>
            </tr><tr>
                <td>Frequency Range:</td><td>[{min_frequency:.3f}, {max_frequency:.3f}] GHz</td>
            </tr></tr>
                <td>Damping:</td><td>{damping:f} &plusmn; {damping_std:f}</td>
            </tr></tr>
                <td>Inhomogeneous Linewidth:</td><td>{deltaH0:f} &plusmn; {deltaH0_std:f} Oe</td>
            </tr></tr>
                <td>Effective Magnetization:</td><td>{Meff:f} &plusmn; {Meff_std:f} Oe</td>
            </tr></tr>
                <td>Gyromagnetic Ratio:</td><td>{gamma:f} &plusmn; {gamma_std:f} GHz/Oe</td>
            </tr></tr>
                <td>Anisotropy Field:</td><td>{Ha:f} &plusmn; {Ha_std:f} Oe</td>
            </tr>
        </table>"""
        return HTML(html.format(
                sample=self.sample,
                min_frequency=self.data['Frequency (GHz)'].min(),
                max_frequency=self.data['Frequency (GHz)'].max(),
                damping=self.damping,
                damping_std=self.damping_std,
                deltaH0=self.linewidth_inhomogeneous,
                deltaH0_std=self.linewidth_inhomogeneous_std,
                Meff=self.effective_magnetization*4*np.pi,
                Meff_std=self.effective_magnetization_std*4*np.pi,
                gamma=self.gyromagnetic_ratio,
                gamma_std=self.gyromagnetic_ratio_std,
                Ha=self.anisotropy_field,
                Ha_std=self.anisotropy_field_std,
               ))

    def plot_linewidth(self, color='b', **kwargs):
        plt.errorbar(
            self.data['Frequency (GHz)'],
            self.data['Linewidth (Oe)'],
            yerr=self.data['Linewidth Std (Oe)'],
            fmt='o', mfc=color, color=color, label=None, **kwargs
        )
        f = np.linspace(0, self.data['Frequency (GHz)'].max()*2)
        plt.plot(
            f,
            linear_linewidth(f, self.gyromagnetic_ratio, self.damping, self.linewidth_inhomogeneous),
            '-', color=color, **kwargs
        )
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Linewidth (Oe)')
        plt.xlim(0, self.data['Frequency (GHz)'].max()*1.1)
        plt.ylim(0, self.data['Linewidth (Oe)'].max()*1.1)
        return plt

    def plot_resonance(self, color='r', **kwargs):
        plt.errorbar(
            self.data['Frequency (GHz)'],
            self.data['Resonance (Oe)']/1000.,
            yerr=self.data['Resonance Std (Oe)']/1000.,
            fmt='o', mfc=color, color=color, label='', **kwargs
        )
        f = np.linspace(0, self.data['Frequency (GHz)'].max()*2)
        fit = inverted_kittel(f, self.gyromagnetic_ratio, self.effective_magnetization, self.anisotropy_field)
        plt.plot(f, fit/1000., '-', color=color, **kwargs)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Resonance (kOe)')
        plt.xlim(0, self.data['Frequency (GHz)'].max()*1.1)
        plt.ylim(fit.min()/1000., (self.data['Resonance (Oe)'].max()/1000.)*1.1)
        return plt

    def plot_overlay(self, frequencies):
        """ Plots a number of dP/dH scans after normalization and resonance
        field subtraction
        """
        pass
