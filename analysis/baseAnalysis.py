import os
from itertools import product
from inspect import getfullargspec

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
from scipy.optimize import leastsq, curve_fit

from pymeasure.experiment import Results, unique_filename

def parse_series_file(direc, series_fname):
    """
    Finds the swept parameter column, all swept series parameters and all
    procedure data files associated with a given series file.

    Assumes that the series file has a particular format:
    - all lines except those with the filenames must begin with ``#``
    - The first line must be of the form ``"# procedure swept column: (value)"``
    - The next lines must be of the form ``"# swept series parameter: (value)"``
    and one must exist for each swept series parameter.
    - All other lines beginning with ``#`` are ignored.


    Parameters
    ----------
    direc : str
        The directory of the series filename.
    series_fname : str
        the filename of the series file.

    Returns
    -------
    series_filenames : list of str
        filenames (names only) of all procedure data files
    procedure_swept_col : str
        name of procedure column swept
    series_swept_params : list of str
        all procedure parameters swept in the series
    """
    with open(os.path.join(direc,series_fname),'r') as f:
        # NOTE: assumes series files adhere to a specific format
        # you should be able to figure it out from the code ;)
        # also enables making custom series files, assuming they all have
        # (nominally) identical swept columns in the procedure.
        line1 = f.readline()
        procedure_swept_col = line1.split(':')[1].strip()
        series_swept_params = []

        # extract all the filenames, representing files from a particular
        # procedure, AND all parameters swept in the series.
        series_filenames = []
        for line in f:
            if line.lower().startswith('# swept series parameter'):
                series_swept_params.append(line.split(':')[1].strip())
            elif line[0] == '#':
                continue
            else:
                series_filenames.append(line.strip())

        return series_filenames, procedure_swept_col, series_swept_params

def load_procedure_files(direc, series_filenames, procedure_swept_col,
                         series_swept_params):
    """
    Loads data from all procedure datafiles, and returns a Dataset containing
    it all. The output of :func:`~.parse_series_file` can be passed (almost)
    directly to this function.

    Parameters
    ----------
    direc : str
        directory containing all of the procedure data files
    series_filenames : list of str
        filenames of all of the procedures
    procedure_swept_col : str
        the data column swept in each procedure
    series_swept_params : list of str
        The procedure parameters swept in the series

    Returns
    -------
    xarray.Dataset
        A Dataset with a data variable for each procedure data column (besides
        the swept one), and dimensions and coordinates corresponding to the
        swept data column and the swept series parameters.
    """

    # load example result
    ex_result = Results.load(os.path.join(direc,series_filenames[0]))

    # record data columns, except one we swept over. Will handle it separately
    data_cols = []
    for col in ex_result.procedure.DATA_COLUMNS:
        if col == procedure_swept_col:
            continue
        data_cols.append(col)
    col_size = ex_result.data[data_cols[0]].size
    new_dims = tuple([procedure_swept_col] + series_swept_params)
    swept_col_data = ex_result.data[procedure_swept_col].values

    # need data_var data to have the correct shape as an array so that
    # all coordinates are taken seriously by the dataset. This exists to
    # reshape the data column into the correct shape
    reshape_helper = [1]*len(series_swept_params)

    # we need to get the ball rolling. Add the first data to growing_ds
    # create data_vars, appropriately reshaped
    new_data_vars = {}
    for col in data_cols:
        new_data_vars[col] = (
            new_dims,
            ex_result.data[col].values.reshape(col_size,*reshape_helper)
        )
    # create new columns, with all from series_swept_params only having one
    # coordinate value (hence the need for reshaping)
    new_coords = {procedure_swept_col: swept_col_data}
    for param in series_swept_params:
        new_coords[param] = [getattr(ex_result.procedure,param)]

    growing_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords)

    # load the rest of the procedure data files
    # if ur lookin here b/c you got an error and were trying to load A SINGLE
    # procedure data file 1. i'm sorry this is implemented poorly 2. ur a dumbass
    for f in series_filenames[1:]:
        rslt = Results.load(os.path.join(direc, f))

        # new data vars (but same dims)
        new_data_vars = {}
        for col in data_cols:
            new_data_vars[col] = (
                new_dims,
                rslt.data[col].values.reshape(col_size,*reshape_helper)
            )
        # new coords (but same dims)
        new_coords = {procedure_swept_col: swept_col_data}
        for param in series_swept_params:
            new_coords[param] = [getattr(rslt.procedure,param)]

        # make dataset corresponding to new procedure data file
        fresh_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords)

        # Now for the magic. of *all* of the ways of combining datasets,
        # the **ONLY** one that does what we want (merging together things
        # with the same dims and data_vars but different coordinates)
        # is the *METHOD* which merges one into the other. It prioritizes
        # the coordinates in the "owner of the method" (fresh_ds in our case)
        # which is why we need to attach growing to this one. Otherwise,
        # if fresh_ds had a new coordinate val in more than one dim, we would
        # get a bunch of nan's as this method attempts to merge the two. This
        # way, even if fresh_ds didn't share coord values in *any* dim with
        # growing_ds since we're prioritizing the ones on fresh_ds so its values
        # will overwrite any nans which may have popped up
        # all these words and it probably wasn't explained clearly. Just play
        # with it in a jupyter notebook smh.
        growing_ds = fresh_ds.combine_first(growing_ds)

    # sort so all coordinates are in a sensible order
    growing_ds = growing_ds.sortby(list(growing_ds.dims))

    return growing_ds # it is now fullly grown :')

def get_coord_selection(ds, drop_dim, gen_empty_ds = True,
                        new_dvar_names = [], **selections):
    """
    Generates a selection of coordinates from a dataset, dropping one dimension.
    Can also generate a new dataset having as coordinates the selected
    ``coords`` and specified ``data_vars``, which are all empty (nonsensical
    entries)

    Parameters
    ----------
    ds : xarray.Dataset
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
        return remaining_dims, coord_combos, remaining_ds, xr.Dataset({})

def make_fit_dataset_guesses(ds, guess_func, param_names, xname,
                      yname, **selections):
    """
    creates a dataset of guesses of param_names given ``guess_func``. To be used
    in :func:`~.fit_dataset`.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing data to fit to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - 1D numpy array of y data
        - numpy array of x data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        All arguments must be accepted, not all must be used.
        Must return a list of guesses to the parameters, in the order given in
        param_names
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by ``guess_func``
    xname : str
        the name of the  ``dim`` of ``ds`` to be fit along
    yname : str
        the name of the ``data_var`` of ``ds`` containing data to be fit to
    **selections
        keywords should be names of ``dims`` of ``ds``
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
    remaining_dims, coord_combos, remaining_ds, guess_ds = get_coord_selection(
        ds,
        xname,
        gen_empty_ds = True,
        new_dvar_names = param_names,
        **selections
    )

    # apply guess_func for each coord combo and record param guesses
    for combo in coord_combos:
        selection_dict = dict(zip(remaining_dims, combo))

        # load x/y data for this coordinate combination
        ydata = remaining_ds[yname].sel(selection_dict).values
        xdata = remaining_ds[xname].values

        # generate guesses
        guesses = guess_func(ydata, xdata, **selection_dict)

        # record fit parameters and their errors
        for i, pname in enumerate(param_names):
            guess_ds[pname].loc[selection_dict] = guesses[i]

    return guess_ds

def fit_dataset(ds, fit_func, guess_func, param_names, xname,
                yname, yerr_name = None, **kwargs):
    """
    Fits values in a dataset to a function. Returns an
    :class:`~.analyzedFit` object

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing data to be fit.
    fit_func : function
        function to fit data to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - 1D numpy array of y data
        - numpy array of x data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        All arguments must be accepted, not all must be used.
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

    if yerr_name is not None and yerr_name not in ds.data_vars:
        raise AttributeError('%s is not a data_var!'%yerr_name)

    selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in ds.dims}
    guesses = make_fit_dataset_guesses(
        ds,
        guess_func,
        param_names,
        xname,
        yname,
        **selections
    )

    # Determine which kwargs can be passed to curve_fit
    cf_argspec = getfullargspec(curve_fit)
    lsq_argspec = getfullargspec(leastsq)
    good_args = cf_argspec.args + lsq_argspec.args
    cf_kwargs = {k: v for k, v in kwargs.items() if k in good_args}

    full_param_names = param_names + [pname+'_err' for pname in param_names]

    # Get the selection and empty fit dataset
    remaining_dims, coord_combos, remaining_ds, fit_ds = get_coord_selection(
        ds,
        xname,
        gen_empty_ds = True,
        new_dvar_names = full_param_names,
        **selections
    )

    # Do the fitting
    for combo in coord_combos:
        selection_dict = dict(zip(remaining_dims, combo))

        # load x/y data for this coordinate combination
        ydata = remaining_ds[yname].sel(selection_dict).values
        xdata = remaining_ds.coords[xname].values

        # load fit parameter guesses for this coordinate combination
        guess = []
        for pname in param_names:
            guess.append(float(guesses[pname].sel(selection_dict).values))

        # load yerr data if given
        if yerr_name is not None:
            yerr = remaining_ds[yerr_name].sel(selection_dict).values
        else:
            yerr = None

        # fit
        popt, pcov = curve_fit(fit_func, xdata, ydata, guess, yerr, **cf_kwargs)
        perr = np.sqrt(np.diag(pcov)) # from curve_fit documentation

        # record fit parameters and their errors
        for i, pname in enumerate(param_names):
            fit_ds[pname].loc[selection_dict] = popt[i]
            fit_ds[pname+'_err'].loc[selection_dict] = perr[i]

    # Create an analyzedFit class to store everything
    return analyzedFit(
        fit_ds,
        remaining_ds,
        fit_func,
        guess_func,
        param_names,
        xname,
        yname,
        yerr_name
    )

def plot_dataset(ds, xname, yname, overlay=False, yerr_name=None,
                 hide_large_errors = False, **kwargs):
    """
    Plots some data in ``ds``.

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

    # TODO: check for xname, yname, yerr_name

    # Get the selections and coordinate combinations
    selections = {dimname: coords for dimname, coords in kwargs.items() if dimname in ds.dims}
    remaining_dims, coord_combos, remaining_ds, _ = get_coord_selection(
        ds,
        xname,
        gen_empty_ds = False,
        **selections
    )

    # Determine which kwargs can be passed to plot
    if yerr_name is None:
        plot_argspec = getfullargspec(Line2D)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
    else:
        ebar_argspec = getfullargspec(plt.errorbar)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

    # Plot for all coordinate combinations
    xdata = remaining_ds.coords[xname].values
    for combo in coord_combos:
        selection_dict = dict(zip(remaining_dims, combo))
        ydata = remaining_ds[yname].sel(selection_dict).values
        label = len(selection_dict.values())*'%g,'%tuple(selection_dict.values())
        if yerr_name is None:
            plt.plot(xdata, ydata, label=label, **plot_kwargs)
        else:
            yerr = remaining_ds[yerr_name].sel(selection_dict).values
            num_pts = yerr.size
            errlims = np.zeros(num_pts).astype(bool)
            if hide_large_errors: # hide outliers if requested
                data_avg = np.mean(np.abs(ydata))
                data_std = np.std(ydata)
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
        plt.ylabel(yname)
        title_str = ''
        for item in selection_dict.items():
            title_str += '%s: %g, '%item
        plt.title(title_str[:-2]) # get rid of trailing comma and space
        if not overlay:
            plt.show()
    if overlay:
        plt.title('Legend: ('
                  +len(selection_dict.keys())*'%s,'%tuple(selection_dict.keys())
                  +')')
        plt.legend()
        plt.show()

class analyzedFit():
    """
    Class containing the results of :func:`~.fit_dataset`.
    Given for convenience to give to plotting functions.

    Parameters
    ----------
    fit_ds : xarray.Dataset
        the dataset which resulted from fitting
    main_ds : xarray.Dataset
        the dataset which was fit over
    fit_func : function
        the function which was used to fit the data
    guess_func : function
        the function which was used to generate initial parameter
        guesses
    param_names : list of str
        names of the parameters of ``fit_func``, in the order it
        accepts them
    xname : str
        the name of the``dim`` of ``main_ds`` which was fit over
    yname : str
        the name of the``data_var`` of ``main_ds`` which was fit over
    yerr_name : str
        Optional. Name of the ``data_var`` of ``main_ds`` used as errors
        on y data.

    Attributes
    ----------
    fit_ds : xarray.Dataset
        the dataset containing all of the found fit parameters and errors
    main_ds : xarray.Dataset
        dataset which was fit over
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
    yname : str
        the name of the :attr:`~.analyzedFit.main_ds`
        ``data_var`` which was fit over
    yerr_name : str
        Optional. the name of the :attr:`~.analyzedFit.main_ds`
        ``data_var`` used as errors on y data.
    """

    def __init__(self, fit_ds, main_ds, fit_func, guess_func, param_names,
                 xname, yname, yerr_name = None):
        """
        Saves variables and extracts ``coords`` and ``dims`` for more convenient
        access.

        Parameters
        ----------
        fit_ds : xarray.Dataset
            the dataset which resulted from fitting
        main_ds : xarray.Dataset
            the dataset which was fit over
        fit_func : function
            the function which was used to fit the data
        guess_func : function
            the function which was used to generate initial parameter
            guesses
        param_names : list of str
            names of the parameters of ``fit_func``, in the order it
            accepts them
        xname : str
            the name of the``dim`` of ``main_ds`` which was fit over
        yname : str
            the name of the``data_var`` of ``main_ds`` which was fit over
        yerr_name : str
            Optional. Name of the ``data_var`` of ``main_ds`` used as errors
            on y data.
        """

        self.fit_ds = fit_ds
        self.coords = self.fit_ds.coords
        self.dims = self.fit_ds.dims
        self.data_vars = self.fit_ds.data_vars
        # QUESTION: should we store parameter guesses?
        self.main_ds = main_ds
        self.fit_func = fit_func
        self.guess_func = guess_func
        self.param_names = param_names
        self.xname = xname
        self.yname = yname
        self.yerr_name = yerr_name

    def plot_fits(self, overlay_data = False, hide_large_errors = True,
                  pts_per_plot = 200, **kwargs):
        """
        Plots the results from fitting of
        :attr:`~analysis.baseAnalysis.analyzedFit.fit_ds`.

        Parameters
        ----------
        overlay_data : bool
            whether to overlay the actual data on top of the
            corresponding fit. Error bars applied if available.
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
            fit_params = [float(selected_ds[param].values) for param
                          in self.param_names]

            # fit the function and plot
            fit_range = self.fit_func(fit_dom, *fit_params)
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

            plt.plot(fit_dom, fit_range)
            # add labels and make the title reflect the current selection
            plt.xlabel(self.xname)
            plt.ylabel(self.yname)
            title_str = ''
            for item in selection_dict.items():
                title_str += '%s: %g, '%item
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.show()

    def plot_params(self, xname, yname, yerr_name = None,
                    hide_large_errors = False, **kwargs):
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
        hide_large_errors : bool
            whether to hide larger errors obstructing the plot.
        **kwargs
            can be:
            - any selections of coordinates of
            :attr:`~.analyzedFit.fit_ds`.
            Keywords should be names of dims, values either single
            coordinate valuesor lists of coordinate values.
            - kwargs passed to ``errorbar``

        Returns
        -------
        None
            Just plots the requested parameters.
        """

        # set yerr_name to default if none given
        if yerr_name is None:
            yerr_name = yname + '_err'

        plot_dataset(self.fit_ds, xname, yname, yerr_name=yerr_name, **kwargs)

class baseAnalysis():
    """
    Class which all other analysis classes should inherit from. Implements
    functions which allow fitting data using arbitrary functions and
    saving/loading of entire datasets to/from netcdf files to avoid overhead.

    Attributes
    ----------
    sweep_ds : xarray.Dataset
        Dataset containing the data
    coords : xarray.coords
        ``coords`` of :attr:`~analysis.baseAnalysis.baseAnalysis.sweep_ds`
    dims : xarray.dims
        ``dims`` of :attr:`~analysis.baseAnalysis.baseAnalysis.sweep_ds`
    data_vars : xarray.data_vars
        ``data_vars`` of :attr:`~analysis.baseAnalysis.baseAnalysis.sweep_ds`

    """

    def __init__(self):
        """ makes a true dummy dataset """

        self.sweep_ds = xr.Dataset({'a': (('b',), [])}, {'b': []})
        self.coords = self.sweep_ds.coords
        self.data_vars = self.sweep_ds.data_vars
        self.dims = self.sweep_ds.dims

    def load_sweep(self, direc, series_file = None, procedure_files = []):
        """
        Loads all of the data from procedure data files into a dataset
        object.

        This general import method depends on procedure_swept_col and
        series_swept_params being defined in ``__init__`` of the child ``*Analysis``
        classes.

        Parameters
        ----------
        direc : str
            The directory the sweep file is in
        series_file : str
            The name of the series file
        procedure_files : list of str
            Any additional procedure files to include.

        Raises
        ------
        ImportError
            if it could not find files to import
        """

        if not procedure_files and series_file is None:
            raise ImportError("Unable to find files to import!")
        if not os.path.isdir(direc):
            raise ImportError("Given directory does not exist!")

        swept_params = []
        all_procedure_files = []

        # import procedure data files from sweep files
        if series_file is not None:
            all_procedure_files, procedure_swept_col, \
             swept_params = parse_series_file(direc, series_file)

        # combine with any explicitly given procedure data files
        all_procedure_files += procedure_files

        # ensure all expected swept params are included
        for param in self.series_swept_params:
            if param not in swept_params:
                swept_params.append(param)

        self.sweep_ds = load_procedure_files(direc, all_procedure_files,
                                                 self.procedure_swept_col,
                                                 swept_params)
        self.coords = self.sweep_ds.coords
        self.dims = self.sweep_ds.dims
        self.data_vars = self.sweep_ds.data_vars

    def load_previous(self, direc, fname, is_sweep_ds = True):
        """
        Loads netCDF containing Datasets.

        Parameters
        ----------
        direc : str
            Directory the netCDF file is in
        fname : str
            Name of the netCDF file
        is_sweep_ds : bool
            If ``True``, puts this dataset into ``sweep_ds``.

        Returns
        -------
        xarray.Dataset
            Dataset contained in the file.
        """

        ds = xr.open_dataset(os.path.join(direc, fname))

        if is_sweep_ds:
            self.sweep_ds = ds
            self.coords = self.sweep_ds.coords
            self.data_vars = self.sweep_ds.data_vars
            self.dims = self.sweep_ds.dims

        return ds

    def save_dataset(self, direc, fname, ext = 'nc', ds = None):
        """
        Saves a dataset as a netCDF file.

        Parameters
        ----------
        direc : str
            Directory to save the netCDF file to
        fname : str
            filename of the netCDF file
        ext : str
            desired extension of the filename, default "nc"
        ds : xarray.Dataset or None
            Dataset to save. If :code:`None`, saves sweep_ds

        Returns
        -------
        None
        """

        basename = os.path.join(direc, fname)
        filename = '%s.%s'%(basename, ext)

        # if it exists, append a number so we don't overwrite
        if os.path.exists(filename):
            i=1
            while os.path.exists('%s_%d.%s'%(basename, i, ext)):
                i += 1
            filename = '%s_%d.%s'%(basename, i, ext)

        if ds is None:
            self.sweep_ds.to_netcdf(path = filename)
        else:
            ds.to_netcdf(path = filename)
