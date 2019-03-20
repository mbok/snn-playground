from pylab import *

tmax = 1000
dt = 0.5

lapp = 10
tr = array([200, 700]) / dt
T = int(ceil(tmax / dt))
v = zeros(T)


class Neuron:
    def __init__(self, v=-70, u=-14, a=0.02, b=0.2, c=-65, d=8):
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.l = 0

    def update(self, dt):
        if self.v < 35:
            dv = (0.04 * self.v + 5) * self.v + 140 - self.u
            nv = self.v + (dv + self.l) * dt
            du = self.a * (self.b * self.v - self.u)
            nu = self.u + du * dt
            self.v = min(nv, 35)
            self.u = nu
        else:
            self.v = self.c
            self.u = self.u + self.d

    def impulse(self, l):
        self.l = l


neuron1 = Neuron()

# loop over time
for t in arange(T - 1):
    if t % 100 == 0:
        neuron1.impulse(lapp)
    else:
        neuron1.impulse(0)
    v[t] = neuron1.v
    neuron1.update(dt)

v[t + 1] = neuron1.v
figure()
tvec = arange(0, tmax, dt)
plot(tvec, v, 'b', label='Voltage trace')
xlabel('Time [ms]')
ylabel('Membran voltage [mV]')
title('A single qIF neuron')
show()
