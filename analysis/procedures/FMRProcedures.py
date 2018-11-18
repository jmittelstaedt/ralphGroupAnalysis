from pymeasure.experiment import Procedure
from pymeasure.experiment import Parameter, FloatParameter, IntegerParameter


class FMRAnalogTrace(Procedure):

    sample = Parameter('Sample')
    frequency = FloatParameter('RF frequency', units='GHz', default=6.0)
    power = FloatParameter('RF power', units='dBm', default=0.0)
    start_field = FloatParameter('Start field', units='Oe')
    end_field = FloatParameter('End field', units='Oe')
    ac_voltage = FloatParameter('AC Field Voltage', units='V', default=0.1)
    sensitivity = FloatParameter('Sensitivity', units='V', default=200e-6)
    time_constant = FloatParameter('Time Constant', units='s', default=30e-3)
    field_points = IntegerParameter('Field points', default=10000)
    trace_period = FloatParameter('Trace period', units='min', default=10)
    trace_averages = IntegerParameter('Trace averages', default=1)
    calibration_directory = Parameter('Field calibration directory', default="./calibration")
    start_time = Parameter('Start time')

    DATA_COLUMNS = ['Field X (V)', 'Field Y (V)', 'Field (G)', 'X (V)', 'Y (V)', 'dP/dH X (V)', 'dP/dH Y (V)']


class FMRDigitalTrace(Procedure):

    sample = Parameter('Sample')
    frequency = FloatParameter('RF frequency', units='GHz', default=6.0)
    power = FloatParameter('RF power', units='dBm', default=0.0)
    start_field = FloatParameter('Start field', units='Oe')
    end_field = FloatParameter('End field', units='Oe')
    field_points = IntegerParameter('Field points', default=50)
    average_period = FloatParameter('Average period', units='s', default=1)
    point_averages = FloatParameter('Point averages', default=500)
    point_delay = FloatParameter('Point delay', units='s', default=1)
    calibration_directory = Parameter('Field calibration directory', default="./calibration")

    DATA_COLUMNS = ['Current (A)', 'Field X (V)', 'Field X Std (V)', 'Field Y (V)',
                     'Field Y Std (V)','Field (G)', 'Field Std (G)', 'X (V)',
                     'X Std (V)', 'Y (V)', 'Y Std (V)']
