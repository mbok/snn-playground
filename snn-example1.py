from numpy import *
from pylab import *
from typing import *



class SimulationContext:
    def __init__(self, clock, clockDif):
        self.spikes = []
        self.clock = clock
        self.clockDif = clockDif

    def addSpike(self, spike):
        self.spikes.append(spike)


class NeuronTracker:
    def __init__(self):
        self.voltage = []
        self.impulse = []
        self.spikes = []
        self.clock = []

    def track(self, context: SimulationContext, voltage, impulse, spiked):
        self.clock.append(context.clock)
        self.voltage.append(voltage)
        self.impulse.append(impulse)
        self.spikes.append(spiked)


class Membrane:
    def __init__(self, v=-70, u=-14, a=0.02, b=0.2, c=-65, d=8):
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.l = 0

    def update(self, context: SimulationContext, tracker: NeuronTracker):
        tv = self.v
        tl = self.l
        dt = context.clockDif
        spiked = False
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
            spiked = True
        self.l = 0
        if tracker is not None:
            tracker.track(context, tv, tl, spiked)
        return spiked

    def impulse(self, l):
        self.l += l


class Synapse:
    def __init__(self, target, impulse, weight, stdpInterval):
        self.target = target
        self.weight = weight
        self.impulse = impulse
        self.preSpikedAt = -1
        self.stdpInterval = stdpInterval

    def spike(self, context: SimulationContext):
        self.target.impulse(self.impulse * self.weight)
        self.preSpikedAt = context.clock

    def preSynapticUpdate(self, context: SimulationContext, postSpiked):
        if self.preSpikedAt < 0:
            return
        if self.weight < 0:
            if postSpiked and (context.clock - self.preSpikedAt) <= self.stdpInterval:
                self.preSpikedAt = -1
                self.weight -= self.weight / 100.0
            elif context.clock - self.preSpikedAt > self.stdpInterval:
                self.preSpikedAt = -1
                self.weight += (-1 - self.weight) / 100.0
        else:
            if postSpiked and (context.clock - self.preSpikedAt) <= self.stdpInterval:
                self.preSpikedAt = -1
                self.weight += (1 - self.weight) / 100.0
            elif context.clock - self.preSpikedAt > self.stdpInterval:
                self.preSpikedAt = -1
                self.weight -= self.weight / 100.0


class Neuron:
    def __init__(self, membrane, position, tracker):
        self.position = position
        self.membrane = membrane
        self.inputs = []
        self.outputs = []
        self.tracker = tracker

    def update(self, context: SimulationContext):
        spiked = self.membrane.update(context, self.tracker)
        for i in self.inputs:
            i.preSynapticUpdate(context, spiked)
        return spiked

    def impulse(self, l):
        self.membrane.impulse(l)

    def addInput(self, input: Synapse):
        self.inputs.append(input)

    def addOutput(self, output: Synapse):
        self.outputs.append(output)

    def spike(self, context: SimulationContext):
        for o in self.outputs:
            o.spike(context)


class Snn:
    def __init__(self, neurons: Iterable[Neuron]):
        self.clock = 0
        self.neurons = neurons

    def connect(self, source: Neuron, target: Neuron):
        s = Synapse(target, 10, 1, 5)
        source.addOutput(s)
        target.addInput(s)


class Simulation:
    def __init__(self, snns: Iterable[Snn], tickInMs=1):
        self.clock = 0
        self.tickInMs = tickInMs
        self.snns = snns
        self.context = SimulationContext(self.clock, self.tickInMs)

    def nextTick(self):
        self.clock += self.tickInMs
        newContext = SimulationContext(self.clock, self.tickInMs)
        for spike in self.context.spikes:
            spike.spike(newContext)
        self.context = newContext
        for snn in self.snns:
            for n in snn.neurons:
                if n.update(self.context):
                    newContext.addSpike(n)


def plotNeuronTracker(tracker: NeuronTracker, titleStr):
    figure()
    plot(tracker.clock, tracker.voltage, 'r', label='Voltage trace')
    plot(tracker.clock, tracker.impulse, 'g', label='Impulse trace')
    plot(tracker.clock, tracker.spikes, 'b', label='Spikes trace')
    xlabel('Time [ms]')
    ylabel('Misc neuron specs')
    title(titleStr)

snn = Snn([Neuron(Membrane(), (0, 0, 0), NeuronTracker()), Neuron(Membrane(), (10, 10, 10), NeuronTracker())])
snn.connect(snn.neurons[0], snn.neurons[1])

tmax = 1000
dt = 0.5
T = int(ceil(tmax / dt))

sim = Simulation([snn], dt)

# loop over time
for t in arange(T - 1):
    if t < 700:
        snn.neurons[0].membrane.impulse(10)
    sim.nextTick()

# fig = plt.figure()
# ax = mplot3d.Axes3D(fig)
# vis = ax.scatter([100, 200], [200, 120], [300, 77], c='#00ee00')

#
# def animate(i):
#     global vis
#     if (random(1) > 0.5):
#         print("Blues")
#         vis.set_alpha(1)
#     else:
#         print("Greens")
#         vis.set_alpha(0)
#
#
# ani = matplotlib.animation.FuncAnimation(fig, animate,
#                                          frames=2, interval=100, repeat=True)

plotNeuronTracker(snn.neurons[0].tracker, "Neuron 1")
plotNeuronTracker(snn.neurons[1].tracker, "Neuron 2")
show()
