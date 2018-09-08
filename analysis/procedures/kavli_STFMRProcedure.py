from pymeasure.experiment import Procedure
from pymeasure.experiment import FloatParameter, BooleanParameter, Parameter

class kavli_STFMRProcedure(Procedure):
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
    temperature : float
        Temperature set point for the sweep
    control_temp : bool
        Whether the temperature was equilibrated Automatically
    equilibration_time : float
        The time which was waited for the temperature to equilibrate, if handled
        automatically.
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
    current2field_calib_name : str
        Name of the magnet calibration file going from current to field
    field2current_calib_name : str
        Name of the magnet calibration file going from field to current
    """
    
    sample_name = Parameter("Sample Name", default='')

    field_strength_start = FloatParameter("Start Magnetic Field", units="T", default=0.)
    field_strength_end = FloatParameter("End Magnetic Field", units="T", default=0.2)
    field_strength_step = FloatParameter("Magnetic Field Step", units="T", default=0.05)
    field_azimuth = FloatParameter("Magnetic Field Azimuthal Angle", units="deg", default=0.)
    current2field_calib_name = Parameter("Current to Magnetic Field Calibration", default='./current2mag_calib.csv')
    field2current_calib_name = Parameter("Magnetic Field to Current Calibration", default='./mag2current_calib.csv')
    delay = FloatParameter("Delay", units="s", default=0.5)
    field_swap = BooleanParameter("Swap Field", default=True)

    rf_freq = FloatParameter("RF Frequency", units="GHz", default=9.0)
    rf_power = FloatParameter("RF Power", units="dBmW", default=15.0)

    temperature = FloatParameter("temperature Setpoint", units="K", default=20.)
    control_temp = BooleanParameter("Automatically Change Setpoint", default=False)
    equilibration_time = FloatParameter("Temp Equilibration Time", units="s", default=1800.)

    sensitivity = FloatParameter("Lockin Sensitivity", units="V", default=0.01)
    time_constant = FloatParameter("Lockin Time Constant", units="s", default=0.5)

    DATA_COLUMNS = ["field_strength","magnet_current","X","Y","true_angle","real_temperature","elapsed_time"]
