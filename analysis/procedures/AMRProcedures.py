from pymeasure.experiment import Procedure
from pymeasure.experiment import FloatParameter, BooleanParameter, Parameter


class AMRAngProcedure(Procedure):
    """
    Daedalus AMR procedure scanning angle.

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
    wheatsone_R1 : float
        Resistance of resistor R1 in the wheatstone bridge
    wheatsone_R2 : float
        Resistance of resistor R2 in the wheatstone bridge
    wheatsone_R3 : float
        Resistance of resistor R3 in the wheatstone bridge
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

    field_polar = FloatParameter("Magnetic Field Polar Angle", units="deg", default=0.)

    field_azimuth_start = FloatParameter("Start Azimuthal Field", units="deg", default=0.)
    field_azimuth_end = FloatParameter("End Azimuthal Field", units="deg", default=0.1)
    field_azimuth_step = FloatParameter("Azimuthal Field Step", units="deg", default=0.05)

    mag_calib_name = Parameter("Magnet Calibration Filename", default='./proj_field')
    delay = FloatParameter("Delay", units="s", default=0.5)

    field_strength = FloatParameter("Field Strength", units="T", default=0.0)

    sensitivity = FloatParameter("Lockin Sensitivity", units="V", default=0.01)
    time_constant = FloatParameter("Lockin Time Constant", units="s", default=0.5)

    applied_voltage = FloatParameter("Applied Sample Voltage", units='V',default=0.)

    wheatsone_R1 = FloatParameter("Wheatstone R1 Fixed Resistance", units='Ohm', default=1886.)
    wheatsone_R2 = FloatParameter("Wheatstone R2 Variable Resistance", units='Ohm', default=100.)
    wheatsone_R3 = FloatParameter("Wheatstone R3 Fixed Resistance", units="Ohm", default=1936.)

    DATA_COLUMNS = ["X","Y","field_azimuth"]

class AMRFieldProcedure(Procedure):
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
    wheatsone_R1 : float
        Resistance of resistor R1 in the wheatstone bridge
    wheatsone_R2 : float
        Resistance of resistor R2 in the wheatstone bridge
    wheatsone_R3 : float
        Resistance of resistor R3 in the wheatstone bridge
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

    field_polar = FloatParameter("Magnetic Field Polar Angle", units="deg",
                                 default=0.)
    field_azimuth = FloatParameter("Magnetic Field Azimuthal Angle",
                                   units="deg", default=0.)

    mag_calib_name = Parameter("Magnet Calibration Filename",
                               default='./proj_field')
    delay = FloatParameter("Delay", units="s", default=0.5)

    field_strength_start = FloatParameter("Initial Field Strength",
                                          units="T", default=0.0)
    field_strength_end = FloatParameter("Final Field Strength",
                                        units="T", default=0.1)
    field_strength_step = FloatParameter("Field Strength Step",
                                         units="T", default=0.0005)
    field_swap = BooleanParameter("Swap Field", default=True)

    sensitivity = FloatParameter("Lockin Sensitivity", units="V",
                                 default=0.01)
    time_constant = FloatParameter("Lockin Time Constant", units="s",
                                   default=0.5)

    applied_voltage = FloatParameter("Applied Sample Voltage", units='V',
                                     default=0.)

    wheatsone_R1 = FloatParameter("Wheatstone R1 Fixed Resistance",
                                  units='Ohm', default=1886.)
    wheatsone_R2 = FloatParameter("Wheatstone R2 Variable Resistance",
                                  units='Ohm', default=100.)
    wheatsone_R3 = FloatParameter("Wheatstone R3 Fixed Resistance",
                                  units="Ohm", default=1936.)


    DATA_COLUMNS = ["X","Y","field_strength"]

class AMRCryoAngProcedure(Procedure):
    """
    Procedure for taking AMR angle sweep measurements with the Kavli setup
    """

    sample_name = Parameter("Sample Name", default='')

    field_strength = FloatParameter("Start Magnetic Field", units="T", default=0.)
    field_azimuth_start = FloatParameter("Magnetic Field Azimuthal Angle", units="deg", default=0.)
    field_azimuth_end = FloatParameter("Magnetic Field Azimuthal Angle", units="deg", default=0.)
    field_azimuth_step = FloatParameter("Magnetic Field Azimuthal Angle", units="deg", default=0.)
    current2field_calib_name = Parameter("Current to Magnetic Field Calibration", default='./current2mag_calib.csv')
    field2current_calib_name = Parameter("Magnetic Field to Current Calibration", default='./mag2current_calib.csv')

    delay = FloatParameter("Delay", units="s", default=0.5)

    sensitivity = FloatParameter("Lockin Sensitivity", units="V", default=0.01)
    time_constant = FloatParameter("Lockin Time Constant", units="s", default=0.5)

    applied_voltage = FloatParameter("Applied Sample Voltage", units='V',default=0.)

    temperature = FloatParameter("temperature Setpoint", units="K", default=20.)
    control_temp = BooleanParameter("Automatically Change Setpoint", default=False)
    equilibration_time = FloatParameter("Temp Equilibration Time", units="s", default=1800.)

    wheatsone_R1 = FloatParameter("Wheatstone R1 Fixed Resistance", units='Ohm', default=1886.)
    wheatsone_R2 = FloatParameter("Wheatstone R2 Variable Resistance", units='Ohm', default=100.)
    wheatsone_R3 = FloatParameter("Wheatstone R3 Fixed Resistance", units="Ohm", default=1936.)

    DATA_COLUMNS = ["X","Y","field_azimuth", "real_temperature"]

class AMRCryoFieldProcedure(Procedure):
    """
    Procedure for taking AMR field sweep measurements with the Kavli setup
    """

    # TODO: actually write
