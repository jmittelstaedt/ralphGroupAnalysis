from itertools import product
from inspect import getfullargspec

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
from scipy.optimize import leastsq, curve_fit

def get_coord_selection(ds, drop_dim, gen_empty_ds = True, new_dvar_names = [], **selections):
    """
    Generates a selection of coordinates from a dataset, dropping one dimension.
    Can also generate a new dataset having as coordinates the selected
    ``coords`` and specified ``data_vars``, which are all empty

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        base Dataset to generate selection from
    drop_dim : str
        name of the dimension of ds which should be dropped
    gen_empty_ds : bool
        whether to generate an empty dataset with the selected coordinates
    new_dvar_names : list of str
        list of data variable names to put in the empty dataset
    **selections
        keywords should be names of dims of ds, values should either
        be single coordinate values or lists of coordinate values of those dims

    Returns
    -------
    remaining_dims : list of str
        the names of the dimensions which remain
    coord_combos : generator of tuple
        a generator object of all possible combinations of the
        selected coordinates. Each coordinate combination is a tuple.
    remaining_ds : xarray.Dataset
        The dataset with all selections made, including ones to ``drop_dim``
    empty_ds : xarray.Dataset
        Empty dataset. If one was not requested, is a completely empty dataset.
    """
    # Check everything is valid
    if drop_dim not in ds.dims:
        raise AttributeError('%s is not a dim!'%drop_dim)
    for key in selections:
        if key in ds.dims:
            if isinstance(selections[key], (float, int)):
                if selections[key] not in ds.coords[key].values:
                    raise ValueError(
                        'Invalid value given for %s, valid values are'%key,
                        ds.coords[key].values
                        )
                # Must make this into a list with a single element so that the
                # dimensions are still "good" to xarray.
                selections[key] = list([selections[key]])
            else: # assuming it's an iterable
                for val in selections[key]:
                    if val not in ds.coords[key].values:
                        raise ValueError(
                            'Invalid value given for %s, valid values are'%key,
                            ds.coords[key].values
                            )
        else:
            raise AttributeError('%s is not a dim!'%key)

    # find coords which should remain
    remaining_selections = {k: v for k, v in selections.items() if k != drop_dim}
    remaining_coords = ds.drop(drop_dim).sel(remaining_selections).coords
    remaining_dims = tuple(remaining_coords.keys())

    # The whole dataset, but with selections made. Also has the possibility of
    # selecting over drop_dim
    remaining_ds = ds.sel(selections)

    # Making a cartesian product of all of the coord vals to loop over
    coord_vals = [remaining_coords[dim].values for dim in remaining_dims]
    # If only one coord value in selection, would return 0-d array which
    # product can't handle, so convert to length 1 list
    for i, cval in enumerate(coord_vals):
        if cval.size == 1:
            coord_vals[i] = [float(cval)]
    coord_combos = product(*coord_vals)

    if gen_empty_ds:
        # create data variable constructors and make empty dataset
        dims_sizes = tuple(remaining_coords[dimname].size for dimname
                           in remaining_dims)
        new_data_vars = {}
        for dname in new_dvar_names:
            new_data_vars[dname] = (remaining_dims, np.empty(dims_sizes))
        empty_ds = xr.Dataset(
            data_vars = new_data_vars,
            coords = remaining_coords
        )
        return remaining_dims, coord_combos, remaining_ds, empty_ds
    else:
        return remaining_dims, coord_combos, remaining_ds, xr.Dataset()

def make_fit_dataArray_guesses(da, guess_func, param_names, xname, **selections):
    """
    creates a dataset of guesses of param_names given ``guess_func``. To be used
    in :func:`~.fit_dataArray`.

    Parameters
    ----------
    da : xarray.DataArray
        data array containing data to fit to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - numpy array of x data
        - 1D numpy array of y data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        All arguments must be accepted, not all must be used.
        As a hint, if designing for unexpected dims you can include **kwargs at
        the end. This will accept any keword arguments not explicitly defined
        by you and just do nothing with them.
        Must return a list of guesses to the parameters, in the order given in
        param_names
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by ``guess_func``
    xname : str
        the name of the  ``dim`` of ``da`` to be fit along
    yname : str
        the name of the ``data_var`` of ``da`` containing data to be fit to
    **selections
        keywords should be names of ``dims`` of ``da``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        have parameter guesses generated. If no selections given, guesses are
        generated for everything

    Returns
    -------
    xarray.Dataset
        A Dataset with param_names as data_vars containing all guesses, and all
        ``dims`` of ``ds`` besides xname with the same coordinates, unless
        otherwise specified in ``**selections``.
    """

    # Generate coordinate combinations from selection and empty dataset
    remaining_dims, coord_combos, remaining_da, guess_ds = get_coord_selection(
        da,
        xname,
        gen_empty_ds = True,
        new_dvar_names = param_names,
        **selections
    )

    # apply guess_func for each coord combo and record param guesses
    for combo in coord_combos:
        selection_dict = dict(zip(remaining_dims, combo))

        # load x/y data for this coordinate combination
        ydata = remaining_da.sel(selection_dict).values
        xdata = remaining_da[xname].values

        # Deal with any possible spurious data
        if np.all(np.isnan(ydata)):
            # there is no meaningful data. Fill guesses with nan's
            print('Encountered entire nan column at :', selection_dict)
            for i, pname in enumerate(param_names):
                guess_ds[pname].loc[selection_dict] = np.nan
            continue
        else:
            # remove bad datapoints
            good_pts = np.logical_and(np.isfinite(ydata), np.isfinite(xdata))
            xdata = xdata[good_pts]
            ydata = ydata[good_pts]

        # generate guesses
        guesses = guess_func(xdata, ydata, **selection_dict)

        # record fit parameters and their errors
        for i, pname in enumerate(param_names):
            guess_ds[pname].loc[selection_dict] = guesses[i]

    return guess_ds

def fit_dataset(ds, fit_func, guess_func, param_names, xname, yname, yerr_name=None, bootstrap_samples=0, **kwargs):
    """
    Fits values in a dataset to a function. Returns an
    :class:`~.analyzedFit` object. 

    Convenience function which calls :func:`~.fit_dataArray`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing data to be fit.
    fit_func : function
        function to fit data to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - numpy array of x data
        - 1D numpy array of y data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        As a hint, if designing for unexpected dims you can include **kwargs at
        the end. This will accept any keword arguments not explicitly defined
        by you and just do nothing with them.
        All arguments must be accepted, not all must be used.fit results
        Must return a list of guesses to the parameters, in the order given in
        ``param_names``
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by guess_func
    xname : str
        the name of the ``dim`` of ``ds`` to be fit along
    yname : str
        the name of the  containing data to be fit to
    yerr_name : str
        Optional. the name of the ``data_var`` of ``ds`` containing errors in data
        to be fit.
    bootstrap_samples : int
        Number of boostrap samples to draw to get beter statistics on the
        parameters and their errors. If zero no boostrap resampling is done
    **kwargs
        can be:
        - names of ``dims`` of ``ds``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        are fit to . If no selections given, everything is fit to.
        - kwargs of ``curve_fit``

    Returns
    -------
    analyzedFit
        Object containing all results of fitting and some convenience methods.
    """

    if yerr_name is not None:
        if yerr_name not in ds.data_vars:
            raise AttributeError('%s is not a data_var!'%yerr_name)
        else:
            yerr_da = ds[yerr_name]
    else:
        yerr_da=None

    fit_da = ds[yname]

    return fit_dataArray(fit_da, fit_func, guess_func, param_names, xname,
                         yname, yerr_da, bootstrap_samples, **kwargs)

def fit_dataArray(da, fit_func, guess_func, param_names, xname, yname=None, yerr_da=None, bootstrap_samples=0, **kwargs):
    """
    Fits values in a data array to a function. Returns an
    :class:`~.analyzedFit` object

    Parameters
    ----------
    da : xarray.DataArray
        Dataset containing data to be fit.
    fit_func : function
        function to fit data to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - numpy array of x data
        - 1D numpy array of y data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        As a hint, if designing for unexpected dims you can include ``**kwargs``
        at the end. This will accept any keword arguments not explicitly defined
        by you and just do nothing with them.
        All arguments must be accepted, not all must be used. fit results
        Must return a list of guesses to the parameters, in the order given in
        ``param_names``
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by guess_func
    xname : str
        the name of the ``dim`` of ``da`` to be fit along
    yname : str or None
        Optional. The name of the y data being fit over.
    yerr_da : xarray.DataArray or None
        Optional. If provided, must be a data array containing errors in the
        data contained in ``da``. Must have the same coordinates as ``da``
    bootstrap_samples : int
        Number of boostrap samples to draw to get beter statistics on the
        parameters and their errors. If zero no boostrap resampling is done
    **kwargs
        can be:
        - names of ``dims`` of ``da``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        are fit to . If no selections given, everything is fit to.
        - kwargs of ``curve_fit``

    Returns
    -------
    analyzedFit
        Object containing all results of fitting and some convenience methods.
    """

    selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in da.dims}
    guesses = make_fit_dataArray_guesses(
        da,
        guess_func,
        param_names,
        xname,
        **selections
    )
    if yerr_da is not None:
        remaining_yerr_da = yerr_da.sel(selections)
    else:
        remaining_yerr_da = None

    # Determine which kwargs can be passed to curve_fit
    cf_argspec = getfullargspec(curve_fit)
    lsq_argspec = getfullargspec(leastsq)
    good_args = cf_argspec.args + lsq_argspec.args
    cf_kwargs = {k: v for k, v in kwargs.items() if k in good_args}

    full_param_names = param_names + [pname+'_err' for pname in param_names]

    # Get the selection and empty fit dataset
    remaining_dims, coord_combos, remaining_da, fit_ds = get_coord_selection(
        da,
        xname,
        gen_empty_ds = True,
        new_dvar_names = full_param_names,
        **selections
    )

    # Do the fitting
    for combo in coord_combos:
        selection_dict = dict(zip(remaining_dims, combo))

        # load x/y data for this coordinate combination
        ydata = remaining_da.sel(selection_dict).values
        xdata = remaining_da.coords[xname].values
        if yerr_da is not None:
            yerr = remaining_yerr_da.sel(selection_dict).values
        else:
            yerr = None

        # load fit parameter guesses for this coordinate combination
        guess = [float(guesses[pname].sel(selection_dict).values) for pname in param_names]
        guess = np.array(guess)

        # Deal with any possible spurious data
        if np.all(np.isnan(ydata)):
            # there is no meaningful data. Fill fit results with nan's
            print('Encountered entire nan column at : ', selection_dict)
            for i, pname in enumerate(param_names):
                fit_ds[pname].loc[selection_dict] = np.nan
                fit_ds[pname+'_err'].loc[selection_dict] = np.nan
            continue
        else:
            # remove bad datapoints
            good_pts = np.logical_and(np.isfinite(ydata), np.isfinite(xdata))
            if yerr_da is not None:
                good_pts = np.logical_and(good_pts, np.isfinite(yerr))
                yerr = yerr[good_pts]
            xdata = xdata[good_pts]
            ydata = ydata[good_pts]
        
        if ydata.size < len(param_names):
            print("Less finite datapoints than parameters at : ", selection_dict)
            for i, pname in enumerate(param_names):
                fit_ds[pname].loc[selection_dict] = np.nan
                fit_ds[pname+'_err'].loc[selection_dict] = np.nan
            continue

        # fit
        popt, pcov = curve_fit(fit_func, xdata, ydata, guess, yerr, **cf_kwargs)
        perr = np.sqrt(np.diag(pcov)) # from curve_fit documentation

        if bootstrap_samples > 0: # TODO: figure out a better way of handling points with large errors
            fit_ys = fit_func(xdata, *popt)
            residuals = ydata - fit_ys

            if yerr_da is not None:
                ys_weights = 1/yerr**2
                ys_weights = ys_weights / ys_weights.sum()
            else:
                ys_weights = None
            popts = []
            for _ in range(bootstrap_samples):
                # draw with replacement, weighting each residual by 1/error**2 so points
                # with large errors are suppressed
                residual_sample = np.random.choice(residuals, size=ydata.size, replace=True, p=ys_weights)
                y_sample = fit_ys + residual_sample
                popt_samp, _ = curve_fit(fit_func, xdata, y_sample, guess, **cf_kwargs)
                popts.append(popt_samp)
            popts = np.array(popts)
            popt = popts.mean(0)
            perr = popts.std(0)

        # record fit parameters and their errors
        for i, pname in enumerate(param_names):
            fit_ds[pname].loc[selection_dict] = popt[i]
            fit_ds[pname+'_err'].loc[selection_dict] = perr[i]

    return analyzedFit(
        fit_ds,
        remaining_da,
        fit_func,
        guess_func,
        param_names,
        xname,
        yname,
        remaining_yerr_da
    )

def plot_dataset(ds, xname, yname, overlay=False, yerr_name=None, hide_large_errors=False, show_legend=True, **kwargs):
    """
    Plots some data in ``ds``.

    Convenience function which calls :func:`~.plot_dataArray`

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing data to plot
    xname : str
        name of the ``dim`` of ``ds`` to plot along
    yname : str
        name of the ``data_var`` of ``ds`` containing data to plot
    overlay : bool
        Whether all plots should be overlayed on top of one another on a
        single plot, or if each thing should have its own plot.
    yerr_name : str
        optional. If specified, should be the name of some ``data_var`` of ``ds``
        containing errors in y data.
    hide_large_errors : bool
        If ``True``, errorbars which are large compared to the mean
        of the data will be rendered smaller with arrows to denote these errors
        are only "bounds" on the actual error. Will also move outliers to the
        mean of the data and give them the same error bars as above.
    show_legend : bool
        Whether the legend should be rendered if the plots are overlaid
    **kwargs
        Can either be:
        - names of ``dims`` of ``ds``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        are plotted. If no selections given, everything is plotted.
        - kwargs passed to ``plot`` or ``errorbar``, as appropriate

    Returns
    -------
    None
        Just plots the requested plots.
    """

    yerr_da = None if yerr_name is None else ds[yerr_name]

    plot_datArray(ds[yname], xname, yname, overlay, yerr_da, hide_large_errors,
                  show_legend, **kwargs)

def plot_datArray(da, xname, yname=None, overlay=False, yerr_da=None, hide_large_errors=False, show_legend=True, **kwargs):
    """
    Plots some data in ``da``.

    Parameters
    ----------
    da : xarray.DataArray
        Data array containing data to plot
    xname : str
        name of the ``dim`` of ``da`` to plot along
    yname : str
        Optional. Name of y data being plotted. Will be y label of plots.
    overlay : bool
        Whether all plots should be overlayed on top of one another on a
        single plot, or if each thing should have its own plot.
    yerr_da : xarray.DataArray
        optional. Data array with same coordinates as ``da`` which has data
        to be used for plotting errorbars
    hide_large_errors : bool
        If ``True``, errorbars which are large compared to the mean
        of the data will be rendered smaller with arrows to denote these errors
        are only "bounds" on the actual error. Will also move outliers to the
        mean of the data and give them the same error bars as above.
    show_legend : bool
        Whether the legend should be rendered if the plots are overlaid
    **kwargs
        Can either be:
        - names of ``dims`` of ``da``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        are plotted. If no selections given, everything is plotted.
        - kwargs passed to ``plot`` or ``errorbar``, as appropriate

    Returns
    -------
    None
        Just plots the requested plots.
    """

    # Get the selections and coordinate combinations
    selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in da.dims}
    remaining_dims, coord_combos, remaining_da, _ = get_coord_selection(
        da,
        xname,
        gen_empty_ds = False,
        **selections
    )
    if yerr_da is not None:
        remaining_yerr_da = yerr_da.sel(selections)

    # Determine which kwargs can be passed to plot
    if yerr_da is None:
        plot_argspec = getfullargspec(Line2D)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
    else:
        ebar_argspec = getfullargspec(plt.errorbar)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

    # Plot for all coordinate combinations
    xdata = remaining_da.coords[xname].values
    for combo in coord_combos:
        selection_dict = dict(zip(remaining_dims, combo))
        ydata = remaining_da.sel(selection_dict).values
        # don't plot if there are only nan's
        if np.all(np.isnan(ydata)) == True:
            continue
        label = (len(selection_dict.values())*'{},').format(*tuple(selection_dict.values()))
        label = label[:-1] # remove trailing comma
        if yerr_da is None:
            plt.plot(xdata, ydata, label=label, **plot_kwargs)
        else:
            yerr = remaining_yerr_da.sel(selection_dict).values
            num_pts = yerr.size
            errlims = np.zeros(num_pts).astype(bool)
            if hide_large_errors: # hide outliers if requested
                data_avg = np.nanmean(np.abs(ydata))
                data_std = np.nanstd(ydata)
                for i, err in enumerate(yerr): 
                    if err > 5*data_std:
                        yerr[i] = data_std*.5 # TODO: Find some better way of marking this
                        errlims[i] = True
                for i, val in enumerate(ydata):
                    if np.abs(val - data_avg) > 5*data_std:
                        ydata[i] = data_avg
                        yerr[i] = data_std*0.5
                        errlims[i] = True
            plt.errorbar(xdata, ydata, yerr, lolims=errlims, uplims=errlims,
                         label=label, **plot_kwargs)
        plt.xlabel(xname)
        if yname is not None:
            plt.ylabel(yname)
        title_str = ''
        for item in selection_dict.items():
            title_str += '{}: {}, '.format(*item)
        plt.title(title_str[:-2]) # get rid of trailing comma and space
        if not overlay:
            plt.show()
    if overlay:
        plt.title('%s vs %s'%(yname, xname))
        if show_legend:
            legend_title = len(selection_dict.keys())*'%s,'%tuple(selection_dict.keys())
            legend_title = legend_title[:-1] # remove trailing comma
            plt.legend(title=legend_title)
        plt.show()

def combine_new_ds_dim(ds_dict, new_dim_name):
    """
    Combines a dictionary of datasets along a new dimension using dictionary keys
    as the new coordinates.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray Datasets or dataArrays
    new_dim_name : str
        The name of the newly created dimension

    Returns
    -------
    xarray.Dataset
        Merged Dataset or DataArray

    Raises
    ------
    ValueError
        If the values of the input dictionary were of an unrecognized type
    """

    expanded_dss = []

    for k, v in ds_dict.items():
        expanded_dss.append(v.expand_dims(new_dim_name))
        expanded_dss[-1][new_dim_name] = [k]
    new_ds = xr.concat(expanded_dss, new_dim_name)

    return new_ds

class analyzedFit():
    """
    Class containing the results of :func:`~.fit_dataset`.
    Given for convenience to give to plotting functions.

    Parameters
    ----------
    fit_ds : xarray.Dataset
        the dataset which resulted from fitting
    main_da : xarray.DataArray
        the data array which was fit over
    fit_func : function
        the function which was used to fit the data
    guess_func : function
        the function which was used to generate initial parameter
        guesses
    param_names : list of str
        names of the parameters of ``fit_func``, in the order it
        accepts them
    xname : str
        the name of the``dim`` of ``main_da`` which was fit over
    yname : str or None
        Name of the y data which was fit over
    yerr_da : xarray.dataArray or None
        The data array containing y error information for the data in ``main_da``

    Attributes
    ----------
    fit_ds : xarray.Dataset
        the dataset containing all of the found fit parameters and errors
    main_da : xarray.DataArray
        the data array which was fit over
    coords : xarray.coords
        ``coords`` of :attr:`~.analyzedFit.fit_ds`
    dims : xarray.dims
        ``dims`` of :attr:`~.analyzedFit.fit_ds`
    data_vars : xarray.data_vars
        ``data_vars`` of :attr:`~.analyzedFit.fit_ds`
    fit_func : function
        function which was fit
    guess_func : function
        function used for generating guesses
    param_names : list of str
        Names of parametrs fit to
    xname : str
        name of :attr:`~.analyzedFit.main_ds` ``dim``
        which was fit over
    yname : str or None
        the name of the y data which was fit over
    yerr_da : xarray.dataArray or None
        The data array containing y error information for the data in ``main_da``
    """

    def __init__(self, fit_ds, main_da, fit_func, guess_func, param_names,
                 xname, yname=None, yerr_da=None):
        """
        Saves variables and extracts ``coords`` and ``dims`` for more convenient
        access.

        Parameters
        ----------
        fit_ds : xarray.Dataset
            the dataset which resulted from fitting
        main_da : xarray.DataArray
            the data array which was fit over
        fit_func : function
            the function which was used to fit the data
        guess_func : function
            the function which was used to generate initial parameter
            guesses
        param_names : list of str
            names of the parameters of ``fit_func``, in the order it
            accepts them
        xname : str
            the name of the``dim`` of ``main_da`` which was fit over
        yname : str or None
            the name of the y data which was fit over
        yerr_da : xarray.dataArray or None
            The data array containing y error information for the data in ``main_da``
        """

        self.fit_ds = fit_ds
        self.coords = self.fit_ds.coords
        self.dims = self.fit_ds.dims
        self.data_vars = self.fit_ds.data_vars
        # QUESTION: should we store parameter guesses?
        self.main_da = main_da
        self.fit_func = fit_func
        self.guess_func = guess_func
        self.param_names = param_names
        self.xname = xname
        self.yname = yname
        self.yerr_da = yerr_da

    def plot_fits(self, overlay_data = True, hide_large_errors = True,
                  pts_per_plot = 200, **kwargs):
        """
        Plots the results from fitting of
        :attr:`~baseAnalysis.analyzedFit.fit_ds`.

        Parameters
        ----------
        overlay_data : bool
            whether to overlay the actual data on top of the
            corresponding fit. Error bars applied if available.
        hide_large_errors : bool
            Whether to hide very large error bars
        pts_per_plot : int
            How many points to use for the ``fit_func`` domain.
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
            Just plots the requested fits.
        """

        remaining_dims = list(self.dims.keys())
        selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in self.fit_ds.dims}
        coord_combos = product(
            *[np.array(self.fit_ds.sel(selections).coords[dimname].values, ndmin = 1)
            for dimname in remaining_dims])

        # Generate domain
        fit_dom = np.linspace(self.main_da.coords[self.xname].values.min(),
                              self.main_da.coords[self.xname].values.max(),
                              pts_per_plot)

        # Determine which kwargs can be passed to plot
        if self.yerr_da is None:
            plot_argspec = getfullargspec(Line2D)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
        else:
            ebar_argspec = getfullargspec(plt.errorbar)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

        for combo in coord_combos:
            selection_dict = dict(zip(remaining_dims, combo))
            selected_ds = self.fit_ds.sel(selection_dict)

            # extract the fit parameters
            fit_params = [float(selected_ds[param].values) for param
                          in self.param_names]

            # don't plot if there is no meaningful data
            if np.all(np.isnan(fit_params)):
                continue

            # fit the function and plot
            fit_range = self.fit_func(fit_dom, *fit_params)
            # overlay data if requested
            if overlay_data:
                data_dom = self.main_da.coords[self.xname].values
                data_range = self.main_da.sel(selection_dict).values
                # plot errorbars if available
                if self.yerr_da is not None:
                    yerr = self.yerr_da.sel(selection_dict).values
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

            plt.plot(fit_dom, fit_range)
            # add labels and make the title reflect the current selection
            plt.xlabel(self.xname)
            if self.yname is not None:
                plt.ylabel(self.yname)
            title_str = ''
            for item in selection_dict.items():
                title_str += '{}: {}, '.format(*item)
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.show()

    def plot_params(self, xname, yname, yerr_name = None, **kwargs):
        """
        Plots the fit parameters and their errors vs some dimension. Thin
        wrapper around :func:`~.plot_dataset`.

        Parameters
        ----------
        xname : str
            name of some ``dim`` of
            :attr:`~.analyzedFit.fit_ds`
        yname : str
            name of some ``data_var`` of
            :attr:`~.analyzedFit.fit_ds`
        yerr_name : str
            name of some ``data_var`` of
            :attr:`~.analyzedFit.fit_ds` representing
            error in yname.
            If none, defaults to ``(yname)_err'``
        **kwargs
            passed along to :func:``~.plot_dataset``

        Returns
        -------
        None
            Just plots the requested parameters.
        """

        # set yerr_name to default if none given
        if yerr_name is None:
            yerr_name = yname + '_err'

        plot_dataset(self.fit_ds, xname, yname, yerr_name=yerr_name, **kwargs)

    def fit_params(self, fit_func, guess_func, param_names, xname,
                    yname, ignore_yerr=False, **kwargs):
        """
        Fits the fit parameters in :attr:`~.analyzedFit.fit_ds` to some
        function.

        the y errors are assumed to be stored in the form yname + '_err', as
        this should be the default output of :func:`~.fit_dataset` which is what
        should have made this :class:`~.analyzedFit` object.

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
        xname : str
            the name of the ``dim`` of :attr:`~.analyzedFit.fit_ds` to be fit
            along
        yname : str
            the name of the ``dim`` containing data to be fit to
        ignore_yerr : bool
            whether to ignore yerrs when fitting
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

        yerr_name = None if ignore_yerr else yname+'_err'
        return fit_dataset(self.fit_ds, fit_func, guess_func, param_names, xname,
                           yname, yerr_name, **kwargs)
