import numpy as np

class SimpleAngAccelProfile:
    def __init__(self, 
                 sim_dt: float, 
                 a: float = 200.0, 
                 t0: float = 0.0,
                 t0_t1: float = 0.4,
                 t1_t2: float = 0.2):
        """Simple angular acceleration profile defined as follows;
        alpha(t) = a*(t-t0), for t0 < t < t1
        alpha(t) = a*(t1-t0), for t1 < t < t2
        alpha(t) = 0, otherwise
        
        All variables t are in seconds.
        
        Args:
        - sim_dt: Simulation time-discretization in seconds.
        - a: Angular acceleration in rad/s^2.
        - t0: Start time for acceleration in seconds.
        - t0_t1: Time duration for acceleration in seconds (mathematically: t1-t0).
        - t1_t2: Time for constant angular velocity in seconds (mathematically: t2-t1).
        
        """
        self.sim_dt = sim_dt
        self.acceleration = a
        self.t0 = t0
        self.t1 = t0 + t0_t1
        self.t2 = self.t1 + t1_t2
    
    def get_ang_vel(self, count: int):
        "Returns angular velocity in rad/s at simulation step count."
        current_time = count * self.sim_dt
        if current_time < self.t0:
            return None
        elif current_time < self.t1:
            return self.acceleration * (current_time - self.t0)
        elif current_time < self.t2:
            return self.acceleration * (self.t1 - self.t0)
        else:
            return None
