import math
import numpy as np
import random
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
                self.weight -= self.weight / 10.0
            elif context.clock - self.preSpikedAt > self.stdpInterval:
                self.preSpikedAt = -1
                self.weight += (-1 - self.weight) / 10.0
        else:
            if postSpiked and (context.clock - self.preSpikedAt) <= self.stdpInterval:
                self.preSpikedAt = -1
                self.weight += (1 - self.weight) / 10.0
            elif context.clock - self.preSpikedAt > self.stdpInterval:
                self.preSpikedAt = -1
                self.weight -= self.weight / 10.0


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
        print("Neuron spiking: " + str(self))
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
    def create(neuronCount, synapseMeanPerNeuron=1000, synapseStdDeviationPerNeuron=500):
        snn = Snn()
        spaceSize = math.ceil(math.pow(neuronCount, 1. / 3))
        space = np.empty((spaceSize, spaceSize, spaceSize), dtype=object)
        i = 0
        x = 0
        y = 0
        z = 0
        posDeviation = 0.1
        while i < neuronCount:
            n = Neuron(Membrane(tracker=NeuronTracker()),
                       (x + randn() * posDeviation, y + randn() * posDeviation, z + randn() * posDeviation))
            snn.neurons.append(n)
            space[x, y, z] = n
            x += 1
            if x >= spaceSize:
                y += 1
                x = 0
            if y >= spaceSize:
                z += 1
                y = 0
            i += 1
        totalSynapses = 0
        for x in range(spaceSize):
            for y in range(spaceSize):
                for z in range(spaceSize):
                    if space[x, y, z] is not None:
                        synapsesCount = int(synapseMeanPerNeuron + randn() * synapseStdDeviationPerNeuron)
                        totalSynapses += snn.connectInSpace(space[x, y, z], synapsesCount, space, spaceSize, x, y, z)
        print("Finished SNN generation")
        print("Total neurons: " + str(neuronCount))
        print("Total synapses: " + str(totalSynapses))
        print("Avg synapses per neuron: " + str(totalSynapses / neuronCount))
        return snn

    def connectInSpace(self, neuron, totalCount, space, spaceSize, x, y, z):
        connected = set()
        axonLen = math.ceil(abs(randn() * spaceSize / 3))
        primeAxis = random.randint(0, 2)
        primeDir = random.choice([-1, 1])
        step1 = random.random() * random.choice([-1, 1])
        step2 = random.random() * random.choice([-1, 1])
        origCount = totalCount
        if primeAxis == 0:
            xd = primeDir
            yd = step1
            zd = step2
        elif primeAxis == 1:
            xd = step1
            yd = primeDir
            zd = step2
        else:
            xd = step1
            yd = step2
            zd = primeDir
        while axonLen > 0 and totalCount > 0:
            totalCount -= self.connectNeighboursAt(neuron, connected, int(totalCount / axonLen), space, spaceSize, (
                self.getBoundedIndex(spaceSize, x), self.getBoundedIndex(spaceSize, y),
                self.getBoundedIndex(spaceSize, z)))
            x += xd
            y += yd
            z += zd
            axonLen -= 1
        return origCount - totalCount

    def getBoundedIndex(self, spaceSize, value):
        return max(0, min(spaceSize - 1, math.ceil(value)))

    def connectNeighboursAt(self, source, connected, count, space, spaceSize, atPos):
        origCount = count
        target = space[atPos[0], atPos[1], atPos[2]]
        if source != target and target is not None and target not in connected:
            self.connect(source, target)
            connected.add(target)
            count -= 1
        bounds = [[atPos[0], atPos[0]], [atPos[1], atPos[1]], [atPos[2], atPos[2]]]
        nextMovingAxis = random.randint(0, 2)
        while count > 0 and not np.array_equal(bounds,
                                               [[0, (spaceSize - 1)], [0, (spaceSize - 1)], [0, (spaceSize - 1)]]):
            if random.randint(0, 1) == 1 and bounds[nextMovingAxis][0] > 0:
                bounds[nextMovingAxis][0] -= 1
                axisMovedPos = bounds[nextMovingAxis][0]
            elif bounds[nextMovingAxis][1] < spaceSize - 1:
                bounds[nextMovingAxis][1] += 1
                axisMovedPos = bounds[nextMovingAxis][1]
            else:
                break
            if nextMovingAxis == 0:
                candidates = space[axisMovedPos, bounds[1][0]:bounds[1][1] + 1, bounds[2][0]:bounds[2][1] + 1]
            elif nextMovingAxis == 1:
                candidates = space[bounds[0][0]:bounds[0][1] + 1, axisMovedPos, bounds[2][0]:bounds[2][1] + 1]
            else:
                candidates = space[bounds[0][0]:bounds[0][1] + 1, bounds[1][0]:bounds[1][1] + 1, axisMovedPos]
            # print("Candidates: " + str(bounds)+" / "+ np.array2string(candidates))
            for c1 in range(len(candidates)):
                for c2 in range(len(candidates[c1])):
                    c = candidates[c1, c2]
                    if source != c and c is not None and c not in connected:
                        connected.add(c)
                        self.connect(source, c, weight=min(randn() + 0.9, 1))
                        count -= 1
                    if count <= 0:
                        return origCount - count
            nextMovingAxis = (nextMovingAxis + 1) % 3
        return origCount - count


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
