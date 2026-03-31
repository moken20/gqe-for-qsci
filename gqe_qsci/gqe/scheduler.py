from abc import ABC, abstractmethod

class TemperatureScheduler(ABC):
    @abstractmethod
    def get_inverse_temperature(self) -> float:
        """Return the temperature for the given epoch."""
        
    def update(self, **kwargs):
        """Update the scheduler state if necessary."""
        pass

    def reset(self):
        """Reset scheduler state to its initial temperature."""
        pass
        
class ConstantScheduler(TemperatureScheduler):
    def __init__(self, initial):
        self.initial = initial
    
    def get_inverse_temperature(self) -> float:
        return self.initial
    
    def update(self, **kwargs):
        pass

    def reset(self):
        pass
    

class LinearScheduler(TemperatureScheduler):
    def __init__(self, initial, delta):
        self.initial = initial
        self.delta = delta
        self.current = initial
        self.current_epoch = 0

    def get_inverse_temperature(self) -> float:
        # Linear decay from initial to final temperature
        return self.initial + (self.final - self.initial) * (self.current_epoch / self.num_epochs)

    def update(self, **kwargs):
        self.current_epoch += self.delta

    def reset(self):
        self.current = self.initial
        self.current_epoch = 0
        
        
class VarBasedScheduler(TemperatureScheduler):
    def __init__(self, initial, delta, target_var):
        self.initial = initial
        self.delta = delta
        self.current = initial
        self.target_var = target_var
    
    def get_inverse_temperature(self) -> float:
        return self.current
    
    def update(self, **kwargs):
        current_var = kwargs.get("energies").var().item()
        if current_var > self.target_var:
            self.current += self.delta  # Decrease temperature
        else:
            self.current -= self.delta  # Increase temperature

    def reset(self):
        self.current = self.initial