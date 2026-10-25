"""ME106 training-role interface over the unchanged ME105 predictive graph.

No learned weights or streaming bytes are added. A frozen-correction view keeps
exact forecast values while removing the direct median-head gradient from the
NLL and interval objectives. This is a declared training graph, not a classical
derivative identity or a claim of improved forecasts before evaluation.
"""
from .median_split_sensor_world import MedianSplitForecast
from .protected_sensor_world import ProtectedSensorWorld


class PointPrioritySensorWorld(ProtectedSensorWorld):
    @staticmethod
    def uncertainty_forecast(forecast, trace):
        correction = trace['median_correction']
        # Subtract an exactly-zero primal. Avoid (mu-correction)+correction,
        # which could change the recorded forecast through rounding.
        location = forecast.log_location - (correction - correction.detach())
        return MedianSplitForecast(location, forecast.left_scale, forecast.right_scale)
