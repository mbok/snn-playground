import math
import numpy as np
from numpy.random import randn
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
    def __init__(self, v=-70, u=-14, a=0.02, b=0.2, c=-65, d=8, tracker: NeuronTracker = None):
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.l = 0
        self.tracker = tracker

    def update(self, context: SimulationContext):
        tv = self.v
        tl = self.l
        dt = context.clockDif
        spiked = False
        if self.v < 35:
            dv = (0.04 * self.v + 5) * self.v + 140 - self.u
            nv = self.v + dv * dt + self.l
            du = self.a * (self.b * self.v - self.u)
            nu = self.u + du * dt
            self.v = min(nv, 35)
            self.u = nu
        else:
            self.v = self.c
            self.u = self.u + self.d
            spiked = True
        self.l = 0
        if self.tracker is not None:
            self.tracker.track(context, tv, tl, spiked)
        return spiked

    def impulse(self, context: SimulationContext, l):
        self.l += l


class Synapse:
    def __init__(self, target, impulse, weight, stdpInterval):
        self.target = target
        self.weight = weight
        self.impulse = impulse
        self.preSpikedAt = -1
        self.stdpInterval = stdpInterval

    def spike(self, context: SimulationContext):
        self.target.impulse(context, self.impulse * self.weight)
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
    def __init__(self, membrane, position):
        self.position = position
        self.membrane = membrane
        self.inputs = []
        self.outputs = []

    def update(self, context: SimulationContext):
        spiked = self.membrane.update(context)
        for i in self.inputs:
            i.preSynapticUpdate(context, spiked)
        return spiked

    def impulse(self, context: SimulationContext, l):
        self.membrane.impulse(context, l)

    def addInput(self, input: Synapse):
        self.inputs.append(input)

    def addOutput(self, output: Synapse):
        self.outputs.append(output)

    def spike(self, context: SimulationContext):
        for o in self.outputs:
            o.spike(context)


class Snn:
    def __init__(self, neurons: Iterable[Neuron] = []):
        self.neurons = neurons

    def connect(self, source: Neuron, target: Neuron, weight=1.0):
        s = Synapse(target, 10, weight, 5)
        source.addOutput(s)
        target.addInput(s)

    @staticmethod
    def create(neuronCount, synapseMean=1000, synapseStdDeviation=500):
        snn = Snn()
        roomSize = math.ceil(math.pow(neuronCount, 1. / 3))
        room = np.empty((roomSize, roomSize, roomSize), dtype=object)
        i = 0
        x = 0
        y = 0
        z = 0
        posDeviation = 0.25
        while i < neuronCount:
            n = Neuron(Membrane(tracker=NeuronTracker()),
                       (x + randn(1)[0] * posDeviation, y + randn(1)[0] * posDeviation, z + randn(1)[0] * posDeviation))
            snn.neurons.append(n)
            room[x, y, z] = n
            x += 1
            if x >= roomSize:
                y += 1
                x = 0
            if y >= roomSize:
                z += 1
                y = 0
            i += 1
        print("Room: " + np.array2string(room))
        synapsesCount = 1000 + randn(1)[0] * 500
        print("Synapses count: " + str(synapsesCount))
        return snn


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
