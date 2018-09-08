from pymeasure.experiment import Procedure
from pymeasure.experiment import FloatParameter, BooleanParameter, Parameter

class daedalus_STFMRProcedure(Procedure):
    """
    Daedalus STFMR procedure scanning field strength.
    
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
    rf_freq : float
        Frequency of the RF current
    rf_power : float
        Power of the RF current
    field_swap : bool
        Whether we swapped the field direction, to record both positive and 
        negative field strength
    sample_name : str
        Name of the sampe
    DATA_COLUMNS : list of str
        Names of the data taken
    delay : float
        Delay between taking data points in seconds
    sensitivity : float
        Sensitivity of the lockin in volts
    time_constant : float
        Time constant of the lockin in seconds
    mag_calib_name : str
        Name of the magnet calibration file.
    field_polar : float
        polar field angle of the field. Zero is in-plane in degrees
    """

    sample_name = Parameter("Sample Name", default='')

    field_azimuth = FloatParameter("Magnetic Field Azimuthal Angle", units="deg", default=0.)
    field_polar = FloatParameter("Magnetic Field Polar Angle", units="deg", default=0.)

    field_strength_start = FloatParameter("Start Magnetic Field", units="T", default=0.)
    field_strength_end = FloatParameter("End Magnetic Field", units="T", default=0.1)
    field_strength_step = FloatParameter("Magnetic Field Step", units="T", default=0.05)
    mag_calib_name = Parameter("Magnet Calibration Filename", default='./proj_field')
    delay = FloatParameter("Delay", units="s", default=0.5)
    field_swap = BooleanParameter("Swap Field", default=True)

    rf_freq = FloatParameter("RF Frequency", units="GHz", default=12.0)
    rf_power = FloatParameter("RF Power", units="dBmW", default=18.0)

    sensitivity = FloatParameter("Lockin Sensitivity", units="V", default=0.01)
    time_constant = FloatParameter("Lockin Time Constant", units="s", default=0.5)

    DATA_COLUMNS = ["X","Y","field_strength"]
