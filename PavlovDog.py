import nengo
import numpy as np

class Wijeekoon():
    """Wijekoon neuron model. """

    probeable = {
        "spikes",
        "voltage",
        "recovery"
    }

    #reset_voltage = NumberParam("reset_voltage")
    #reset_recovery = NumberParam("reset_recovery")

    def __init__(
        self,
        reset_voltage=0.3, 
        reset_recovery=1.9*0.06,
    ):
        super().__init__()
        self.reset_voltage = reset_voltage
        self.reset_recovery = reset_recovery

    def rates(self, x, gain, bias):
        J = self.current(x, gain, bias)
        voltage = np.zeros_like(J)
        recovery = np.zeros_like(J)

        return settled_firingrate(
            self.step_math,
            J,
            [
                voltage,
                recovery
            ],
            dt=1e-3,
            settle_time=1e-3,
            sim_time=1e0,
        )

    def step_math(self, dt, J, output, voltage, recovery):
        J = np.maximum(0, J ) * 1e-8

        WL1 = 2.3 / 1
        LW2 = 1/2.3
        WL4 = 1.3 / 22
        WL6 = 1.3 / 18
        WL7 = 1.3 / 14

        V_t = 0.5
        V_th = 1.7

        k = 168e-6
        Cv = 0.1e-8
        Cu = 1e-8
        
        alpha = np.where(voltage >= V_t, np.ones_like(voltage), np.zeros_like(voltage))
        gamma = np.where(recovery >= V_t, np.ones_like(recovery), np.zeros_like(recovery))
        beta = np.array([np.logical_and(i,j) for i, j in zip(alpha, gamma)])
        
        #if V >= U - V_t:
        a = alpha * (0.5*(WL1*(voltage - V_t)**2))
        b = beta  * (0.5*(WL4*(recovery - V_t)**2))
        dV = ((k/Cv) * (a - b + J/k)) 
 
        voltage[:] += dV 

        output[:] = (voltage >= V_th) / dt
        voltage[output > 0] = self.reset_voltage

        a = alpha*(0.5*WL1 * LW2 * WL7 * (voltage - V_t)**2)
        g = gamma*(0.5*WL6 * (recovery - V_t)**2)
        dU = ((k/Cu) * (a - g))

        recovery[:] += dU
        recovery[output > 0] = recovery[output > 0] + self.reset_recovery

class Vourkas:
	def __init__(self, dimension=1, r_init = 245):
		self.r = r_init

	def dr(self, Vm, alpha=1e1, beta=0.010, gamma=0.0001, Vs=1.5, Vr=-1.5):
		if Vm < Vr:
			return -alpha*((Vm - Vr)/(gamma + abs(Vm - Vr)))

		elif Vr <= Vm <= Vs:
			return -beta*Vm

		elif Vs < Vm:
			return -alpha*((Vm - Vs)/(gamma + abs(Vm - Vs)))

	def L(self, r, Lo=5, m=82):
		return Lo*(1-m/r)

	def Rt(self, Lvm, fo=310):
		return fo*((np.exp(2*Lvm))/(Lvm))

	def Gt(self, R):
		return 1/R

	def step(self, t, Vm, R_min=100, R_max=390):	
		self.r += self.dr(Vm)*t
		self.r = np.clip(self.r, R_min, R_max)
			
		L_hat = self.L(self.r)
		R_hat = self.Rt(L_hat)
		G_hat = self.Gt(R_hat)
		out = np.interp(Vm*G_hat,[-0.96e-3,0.96e-3],[-1,1])
        
		return out

memristorFood = Vourkas(r_init=100)
memristorBell = Vourkas(r_init=390)

with nengo.Network() as model:
   
    stimFood = nengo.Node(0)
    stimBell = nengo.Node(0)
    
    memristorNodeFood = nengo.Node(memristorFood.step, size_in=1, size_out=1)
    memristorNodeBell = nengo.Node(memristorBell.step, size_in=1, size_out=1)
    
    Food  =  nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Wijeekoon(),seed=0)
    Bell  =  nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Wijeekoon(),seed=0)
    Salivate=nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Wijeekoon(),seed=0)
    
    nengo.Connection(stimFood, Food, function=lambda x:x)
    nengo.Connection(stimBell, Bell, function=lambda x:x)
   
    nengo.Connection(Food, memristorNodeFood)
    nengo.Connection(Bell, memristorNodeBell)
    
    nengo.Connection(memristorNodeFood, Salivate)
    nengo.Connection(memristorNodeBell, Salivate)
    
    nengo.Connection(Salivate, memristorNodeBell, function=lambda x:np.interp(x,[-2,2],[-1,1]))
    nengo.Connection(Salivate, memristorNodeFood, function=lambda x:np.interp(x,[-2,2],[-1,1]))
    
    
