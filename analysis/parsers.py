import os
import re

import pandas as pd
import numpy as np

def pymeasure_parser(data_fname, swept_column, swept_params, codename_converter={}, column_name_converter={}, comment='#'):
    """
    Parses information out of a pymeasure data file

    Parameters
    ----------
    data_fname : str
        filename
    swept_column : str
        name of data column which was swept
    swept_params : list of str
        The "code" names of the parameters swept
    codename_converter : dict
        Dictionary where keys are the plain names of the parameters in the
        pymeasure procedure, and the value is a tuple whose first element is a
        string of the codename of the parameter and the second is a function to
        cast the value (int, float, bool, str etc.)
    column_name_converter : dict
        Dictionary used to map any pymeasure column names to different column
        names. Keys are the expected original column names, values are the 
        new ones. Will only attempt to rename columns included as keys in the
        converter.
    comment : str
        Character used to represent a commented line

    Returns
    -------
    swept_col_data : numpy.array
        Array containing the swept column data values
    data : dict
        Dictonary where the key is a name of a data column and the value is
        a numpy array of values to use.
    param_values : dict
        A dictonary where the key is the name of a parameter and the value is
        its value for this particular data file
    """

    # read parameter plain names and values from data file
    header_read = False
    parameters = {} # plain name : value
    with open(data_fname) as f:
        while not header_read:
            line = f.readline()
            if line.startswith(comment+'\t'): # is a parameter line
                regex = (comment+"\t(?P<name>[^:]+):\s(?P<value>[^\s]+)"
                         "(?:\s(?P<units>.+))?")
                search = re.search(regex, line)
                if search is None:
                    raise Exception("Error parsing header line {}.".format(line))
                else:
                    parameters[search.group('name')] = search.group("value")
            elif line.startswith(comment):
                pass
            else:
                header_read = True

    # get list of plain names of swept_params to check against
    swept_params_plain = [] # plain names
    # TODO: deal with multiple plain names for the same codename
    plainname_converter = {v[0]: k for k, v in codename_converter.items()}
    for pname in swept_params:
        try:
            swept_params_plain.append(plainname_converter[pname])
        except KeyError:
            raise KeyError("Parameter {} missing from codename converter!".format(pname))

    # convert parameters to dictionary of codenames : converted values
    param_values = {}
    for swept_param in swept_params_plain:
        try:
            v = parameters[swept_param]
            codename = codename_converter[swept_param][0]
            converted_val = codename_converter[swept_param][1](v)
            param_values[codename] = converted_val
        except KeyError: # if we do not find the swept param in the parameters dict
            raise ValueError(f"Did not find {swept_param} parameter in the data file!")

    # retrieving procedure data
    data_df = pd.read_csv(data_fname, comment='#')
    data_dict = data_df.to_dict('list')
    new_data_keys = {}
    for k in data_dict.keys():
        if k in column_name_converter:
            new_data_keys[k] = column_name_converter[k]
        else:
            new_data_keys[k] = k
    data_dict = {new_data_keys[k]: np.array(v) for k, v in data_dict.items()}

    # separate swept column data from the non-swept columns
    try:
        swept_col_data = data_dict.pop(swept_column)
    except KeyError:
        raise KeyError(f"{swept_column} was not found in the data file columns!")

    return swept_col_data, data_dict, param_values

def FMR_parser(data_fname, swept_column, swept_params, codename_converter={}, comment='#'):
    """
    Parses information out of a pymeasure data file without using Results.load()
    For particular use with classes from Colin to get nominal field points in
    FMR scans.

    Parameters
    ----------
    data_fname : str
        filename
    swept_column : str
        name of data column which was swept. Not used in this instance, but
        needs to be accepted so callers function properly
    swept_params : list of str
        The names of the parameters swept
    codename_converter : dict
        Dictionary where keys are the plain names of the parameters in the
        pymeasure procedure, and the value is a tuple whose first element is a
        string of the codename of the parameter and the second is a function to
        cast the value (int, float, bool, str etc.)
    comment : str
        Character used to represswept_colent a commented line

    Returns
    -------
    swept_col_data : numpy.array
        Array containing the swept column data values
    data : dict
        Dictonary where the key is a name of a data column and the value is
        a numpy array of values to use.
    param_values : dict
        A dictonary where the key is the name of a parameter and the value is
        its value for this particular data file
    """
    # read parameter plain names and values from data file
    header_read = False
    parameters = {} # plain name to value found
    with open(data_fname) as f:
        while not header_read:
            line = f.readline()
            if line.startswith(comment+'\t'): # is a parameter line
                regex = (comment+"\t(?P<name>[^:]+):\s(?P<value>[^\s]+)"
                         "(?:\s(?P<units>.+))?")
                search = re.search(regex, line)
                if search is None:
                    raise Exception("Error parsing header line {}.".format(line))
                else:
                    parameters[search.group('name')] = search.group("value")
            elif line.startswith(comment):
                pass
            else:
                header_read = True

    # get list of plain names of swept_params to check against
    swept_params_plain = [] # plain names
    plainname_converter = {v[0]: k for k, v in codename_converter.items()}
    for pname in swept_params:
        try:
            swept_params_plain.append(plainname_converter[pname])
        except KeyError:
            raise KeyError("Parameter {} missing from codename converter!".format(pname))

    # convert parameters to dictionary of codenames
    param_values = {}
    for k, v in parameters.items():
        if k in swept_params_plain:
            codename = codename_converter[k][0]
            converted_val = codename_converter[k][1](v)
            param_values[codename] = converted_val
        # don't care if we have extra (unused) parameters in the data file

    # check if values for all swept params were found
    for pname in swept_params:
        if pname not in param_values.keys():
            raise ValueError("Did not find {} parameter value in the data file!".format(pname))

    # retrieving procedure data
    data_df = pd.read_csv(data_fname, comment='#')
    data_dict = data_df.to_dict('list') # converts to a dictionary
    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    # Don't delete any columns since we will make a column of nominal field
    # points.

    # make our own swept column
    field_params = {}
    for k, v in parameters.items():
        if codename_converter[k][0] == 'start_field':
            field_params['field_start'] = codename_converter[k][1](v)
        elif codename_converter[k][0] == 'end_field':
            field_params['field_stop'] = codename_converter[k][1](v)
        elif codename_converter[k][0] == 'field_points':
            field_params['num_field_points'] = codename_converter[k][1](v)
        else:
            pass
    field_points_nominal = np.linspace(field_params['field_start'],
                                       field_params['field_stop'],
                                       field_params['num_field_points'])

    return field_points_nominal, data_dict, param_values

def extract_parameters(fname, param_codes=[], codename_converter={}, comment='#'):
    """
    Parses pymeasure data files and extracts particular parameter values.

    Parameters
    ----------
    fname : str
        filename
    param_codes : list of str
        The codenames of the parameters whose we wish to extract
    codename_converter : dict
        Dictionary where keys are the plain names of the parameters in the
        pymeasure procedure, and the value is a tuple whose first element is a
        string of the codename of the parameter and the second is a function to
        cast the value (int, float, bool, str etc.)
    comment : str
        Character used to represent a commented line

    Returns
    -------
    dict
        A dictionary mapping parameter codenames to their values
    """
    # plainnames of params which we need
    needed_param_plainnames = [k for k, v in codename_converter.items() if v[0] in param_codes]

    # read header and extract needed parameter values
    header_read = False
    parameters = {}
    with open(fname) as f:
        while not header_read:
            line = f.readline()
            if line.startswith(comment+'\t'): # is a parameter line
                regex = (comment+"\t(?P<name>[^:]+):\s(?P<value>[^\s]+)"
                         "(?:\s(?P<units>.+))?")
                search = re.search(regex, line)
                if search is None:
                    raise Exception("Error parsing header line {}.".format(line))
                else:
                    if search.group('name') in needed_param_plainnames:
                        converter = codename_converter[search.group('name')]
                        parameters[converter[0]] = converter[1](search.group("value"))
            elif line.startswith(comment):
                pass
            else:
                header_read = True
    return parameters
