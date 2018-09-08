from pymeasure.experiment import Procedure
from pymeasure.experiment import FloatParameter, BooleanParameter, Parameter


class Hall2HarmAngProcedure(Procedure):
    """
    Daedalus second harmonic Hall procedure scanning angle.
    
    Attributes
    ----------
    field_azimuth_start : float
        Starting field azimuthal angle in degrees
    field_azimuth_end : float
        Final field azimuthal angle in degrees
    field_azimuth_step : float
        Step size of the azimuthal field angle in degrees
    field_strength : float
        Strength of the magnetic field during the scan in Tesla
    applied_voltage : float
        Voltage applied across the wheatstone bridge
    sample_name : str
        Name of the sampe
    DATA_COLUMNS : list of str
        Names of the data taken
    delay : float
        Delay between taking data points in seconds
    sensitivity1 : float
        Sensitivity of the lockin measuring the first harmonic in volts
    time_constant1 : float
        Time constant of the lockin measuring the first harmonic in seconds
    sensitivity2 : float
        Sensitivity of the lockin measuring the second harmonic in volts
    time_constant2 : float
        Time constant of the lockin measuring the second harmonic in seconds
    mag_calib_name : str
        Name of the magnet calibration file.
    field_polar : float
        polar field angle of the field. Zero is in-plane in degrees
    """

    sample_name = Parameter("Sample Name", default='')

    field_polar = FloatParameter("Magnetic Field Polar Angle", units="deg", default=0.)

    field_azimuth_start = FloatParameter("Start Azimuthal Field", units="deg", default=0.)
    field_azimuth_end = FloatParameter("End Azimuthal Field", units="deg", default=0.1)
    field_azimuth_step = FloatParameter("Azimuthal Field Step", units="deg", default=0.05)

    mag_calib_name = Parameter("Magnet Calibration Filename", default='./proj_field')
    delay = FloatParameter("Delay", units="s", default=0.5)

    field_strength = FloatParameter("Field Strength", units="T", default=0.0)

    sensitivity1 = FloatParameter("First Harmonic Lockin Sensitivity", units="V", default=0.01)
    time_constant1 = FloatParameter("First Harmonic Lockin Time Constant", units="s", default=0.5)
    sensitivity2 = FloatParameter("Second Harmonic Lockin Sensitivity", units="V", default=0.01)
    time_constant2 = FloatParameter("Second Harmonic Lockin Time Constant", units="s", default=0.5)

    applied_voltage = FloatParameter("Applied Sample Voltage", units='V',default=0.)

    DATA_COLUMNS = ["X1", "X2", "Y1", "Y2", "field_azimuth"]

class Hall2HarmFieldProcedure(Procedure):
    """
    Daedalus AMR procedure scanning field strength.
    
    Attributes
    ----------
    field_strength_start : float
        Starting field strength in Tesla
    field_strength_end : float
        Final field strength in Tesla
    field_strength_step : float
        Step size of the field strength in Tesla
    field_azimuth : float
        Azimuthal field angle during the sweep
    applied_voltage : float
        Voltage applied across the wheatstone bridge
    field_swap : bool
        Whether we swapped the field direction, to record both positive and 
        negative field strength
    sample_name : str
        Name of the sampe
    DATA_COLUMNS : list of str
        Names of the data taken
    delay : float
        Delay between taking data points in seconds
    sensitivity1 : float
        Sensitivity of the lockin measuring the first harmonic in volts
    time_constant1 : float
        Time constant of the lockin measuring the first harmonic in seconds
    sensitivity2 : float
        Sensitivity of the lockin measuring the second harmonic in volts
    time_constant2 : float
        Time constant of the lockin measuring the second harmonic in seconds
    mag_calib_name : str
        Name of the magnet calibration file.
    field_polar : float
        polar field angle of the field. Zero is in-plane in degrees
    """

    sample_name = Parameter("Sample Name", default='')

    field_polar = FloatParameter("Magnetic Field Polar Angle", units="deg", default=0.)
    field_azimuth = FloatParameter("Magnetic Field Azimuthal Angle", units="deg", default=0.)

    mag_calib_name = Parameter("Magnet Calibration Filename", default='./proj_field')
    delay = FloatParameter("Delay", units="s", default=0.5)

    field_strength_start = FloatParameter("Field Strength", units="T", default=0.0)
    field_strength_end = FloatParameter("Final Field Strength", units="T", default=0.1)
    field_strength_step = FloatParameter("Field Strength Step", units="T", default=0.01)
    field_swap = BooleanParameter("Swap Field", default=True)

    sensitivity1 = FloatParameter("First Harmonic Lockin Sensitivity", units="V", default=0.01)
    time_constant1 = FloatParameter("First Harmonic Lockin Time Constant", units="s", default=0.5)
    sensitivity2 = FloatParameter("Second Harmonic Lockin Sensitivity", units="V", default=0.01)
    time_constant2 = FloatParameter("Second Harmonic Lockin Time Constant", units="s", default=0.5)

    applied_voltage = FloatParameter("Applied Sample Voltage", units='V',default=0.)

    DATA_COLUMNS = ["X1", "X2", "Y1", "Y2", "field_strength"]
