import os
from itertools import product
from inspect import getfullargspec

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr

from .parsers import pymeasure_parser
from .dataset_manipulation import combine_new_ds_dim, analyzedFit

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
    data_fnames : list of str
        filenames (names only) of all procedure data files
    procedure_swept_col : str
        name of procedure column swept
    series_swept_params : list of str
        all procedure parameters swept in the series
    """
    with open(os.path.join(direc,series_fname),'r') as f:
        line1 = f.readline()
        procedure_swept_col = line1.split(':')[1].strip()
        series_swept_params = []

        # extract all the filenames, representing files from a particular
        # procedure, AND all parameters swept in the series.
        data_fnames = []
        for line in f:
            if line.lower().startswith('# swept series parameter'):
                series_swept_params.append(line.split(':')[1].strip())
            elif line.startswith('#') or not line.strip():
                continue
            else:
                data_fnames.append(line.strip())

        return data_fnames, procedure_swept_col, series_swept_params

def load_procedure_files(direc, data_fnames, procedure_swept_col, series_swept_params, parser, **kwargs):
    """
    Loads data from all data files, and returns a Dataset containing
    it all. The output of :func:`~.parse_series_file` can be passed (almost)
    directly to this function.

    Parameters
    ----------
    direc : str
        directory containing all of the procedure data files
    data_fnames : list of str
        filenames of all of the procedures
    procedure_swept_col : str
        the data column swept in each procedure
    series_swept_params : list of str
        The procedure parameters swept in the series
    parser : func
        Function which accepts a data filename, a string representing
        the swept data column, a list of strings representing swept parameters
        in the series of sweeps and optional keyword arguments. It must return
        a numpy array of the swept column data, a dictionary of 1d numpy arrays
        of each of the other data columns, and a dictionary of the values of the
        parameters which were swept in the series.
    kwargs
        Keyword arguments passed along to the ``parser`` function

    Returns
    -------
    xarray.Dataset
        A Dataset with a data variable for each procedure data column (besides
        the swept one), and dimensions and coordinates corresponding to the
        swept data column and the swept series parameters.
    """
    # load initial data
    swept_col_data, data, param_values = parser(os.path.join(direc, data_fnames[0]),
                                                procedure_swept_col,
                                                series_swept_params, **kwargs)
    col_size = swept_col_data.size

    # Make tuple of new dimension names
    new_dims = tuple([procedure_swept_col] + series_swept_params)

    # need data_var data to have the correct shape as an array so that
    # all coordinates are taken seriously by the dataset. This exists to
    # reshape the data column into the correct shape
    reshape_helper = [1]*len(series_swept_params)

    # Create an empty Dataset which we will add all of our actual data to
    growing_ds = xr.Dataset()

    # load the rest of the procedure data files using the same method as above
    for f in data_fnames:
        swept_col_data, data, param_values = parser(os.path.join(direc, f),
                                                    procedure_swept_col,
                                                    series_swept_params,
                                                    **kwargs)

        # Add data to the data variables, appropriately reshaped
        new_data_vars = {}
        for col, data_vals in data.items():
            new_data_vars[col] = (
                new_dims,
                data_vals.reshape(col_size, *reshape_helper)
            )

        # Create coordinate dictionary, putting all single value coordinates
        # into lists of size 1 so xarray is happy.
        new_coords = {k: [v] for k, v in param_values.items()}
        new_coords[procedure_swept_col] = swept_col_data

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

    return growing_ds

def combine_new_dim(ds_dict, new_dim_name):
    """
    Combines a dictionary of datasets along a new dimension using dictionary keys
    as the new coordinates. Inherits functionality from
    :func:`~.dataset_manipulation.combine_new_ds_dim`.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray Datasets or instances of some subclass of
        :class:`~.analyzedFit` or :class:`~.baseAnalysis`
    new_dim_name : str
        The name of the newly created dimension

    Returns
    -------
    xarray.Dataset
        Merged instance of whatever objects were input, with the other
        meta-parameters of that object staying the same.

    Raises
    ------
    ValueError
        If the values of the input dictionary were of an unrecognized type
    """

    test_element = list(ds_dict.values())[0]

    if isinstance(test_element, baseAnalysis):
        sweep_dss = {k: v.sweep_ds for k, v in ds_dict.items()}
        combined_sweep_ds = combine_new_ds_dim(sweep_dss, new_dim_name)
        new_obj = test_element.__class__()
        new_obj.procedure_swept_col = test_element.procedure_swept_col
        new_obj.series_swept_params = test_element.series_swept_params
        new_obj.parser = test_element.parser
        new_obj.codename_converter = test_element.codename_converter
        new_obj.sweep_ds = combined_sweep_ds
        new_obj.coords = new_obj.sweep_ds.coords
        new_obj.data_vars = new_obj.sweep_ds.data_vars
        new_obj.dims = new_obj.sweep_ds.dims
    elif isinstance(test_element, analyzedFit):
        fit_dss = {k: v.fit_ds for k, v in ds_dict.items()}
        main_dss = {k: v.main_ds for k, v in ds_dict.items()}

        new_main_da = combine_new_ds_dim(main_dss, new_dim_name)
        new_fit_ds = combine_new_ds_dim(fit_dss, new_dim_name)
        new_obj = test_element.__class__(new_fit_ds, new_main_da,
                                         test_element.fit_func,
                                         test_element.guess_func,
                                         test_element.param_names,
                                         test_element.xname,
                                         test_element.yname,
                                         test_element.yerr_da)
    elif isinstance(test_element, xr.Dataset) or isinstance(test_element, xr.DataArray):
        new_obj = combine_new_ds_dim(ds_dict, new_dim_name)
    else:
        raise ValueError("""Dictionary values were not of a known type. Values
                         must either inherit from baseAnalysis, analyzedFit
                         or be xarray Datasets or DataArrays""")

    return new_obj

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
        ``coords`` of :attr:`~baseAnalysis.baseAnalysis.sweep_ds`
    dims : xarray.dims
        ``dims`` of :attr:`~baseAnalysis.baseAnalysis.sweep_ds`
    data_vars : xarray.data_vars
        ``data_vars`` of :attr:`~baseAnalysis.baseAnalysis.sweep_ds`
    codename_converter : dict
        Dictionary to convert plain names in pymeasure data files into the
        variable used in the code and a type to convert the data value to.
    """

    def __init__(self):
        self.sweep_ds = xr.Dataset()
        self.coords = self.sweep_ds.coords
        self.data_vars = self.sweep_ds.data_vars
        self.dims = self.sweep_ds.dims
        self.codename_converter = None
        self.parser = pymeasure_parser
        self.procedure_swept_col = None
        self.series_swept_params = [None]

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
        series_file : str or list of str
            The name of the series file(s)
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

        all_swept_params = []
        all_procedure_files = []

        # import procedure data files from sweep filesall_swept_params
        if isinstance(series_file, str):
            all_procedure_files, procedure_swept_col, \
             all_swept_params = parse_series_file(direc, series_file)
        elif isinstance(series_file, list):
            for sfile in series_file:
                auto_procedure_files, procedure_swept_col, \
                 swept_params = parse_series_file(direc, sfile)
                all_procedure_files += auto_procedure_files
                all_swept_params += swept_params
        else: # Assumed none given
            pass

        # combine with any explicitly given procedure data files
        all_procedure_files += procedure_files

        # make all are unique
        all_swept_params = list(set(all_swept_params))
        all_procedure_files = list(set(all_procedure_files))

        # ensure all expected swept params are included
        for param in self.series_swept_params:
            if param not in all_swept_params:
                all_swept_params.append(param)

        self.sweep_ds = load_procedure_files(direc, all_procedure_files,
                                                 self.procedure_swept_col,
                                                 all_swept_params,
                                                 self.parser,
                                                 codename_converter=self.codename_converter)
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
