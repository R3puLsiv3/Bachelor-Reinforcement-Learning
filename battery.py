
class Battery:
    def __init__(self, soc=1.0, charge_efficiency=0.9, discharge_efficiency=0.9):
        self.soc = soc
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

    def calculate_soc(self, action):
        pass
