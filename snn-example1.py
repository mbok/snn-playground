import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
from mpl_toolkits import mplot3d
from numpy import *
from pylab import *

from model import *


def plotNeuronTracker(tracker: NeuronTracker, titleStr):
    figure()
    plot(tracker.clock, tracker.voltage, 'r', label='Voltage trace')
    plot(tracker.clock, tracker.impulse, 'g', label='Impulse trace')
    plot(tracker.clock, tracker.spikes, 'b', label='Spikes trace')
    xlabel('Time [ms]')
    ylabel('Misc neuron specs')
    title(titleStr)


snn = Snn([
    Neuron(Membrane(tracker=NeuronTracker()), (0, 0, 0)),
    Neuron(Membrane(tracker=NeuronTracker()), (10, 10, 10)),
    Neuron(Membrane(tracker=NeuronTracker()), (15, 15, 15)),
    Neuron(Membrane(tracker=NeuronTracker()), (25, 25, 25)),
])
snn.connect(snn.neurons[0], snn.neurons[3], weight=1)
snn.connect(snn.neurons[1], snn.neurons[3], weight=1)
snn.connect(snn.neurons[2], snn.neurons[3], weight=1)

snn = Snn.create(270, synapseMeanPerNeuron=10, synapseStdDeviationPerNeuron=5)

tmax = 1000
dt = 0.5
T = int(ceil(tmax / dt)) * 0

sim = Simulation([snn], dt)

print("Start simulation")
# loop over time
for t in arange(T - 1):
    if t < 700 and random.random() < 0.75:
        sim.context.addSpike(snn.neurons[0])
    if t < 700 and random.random() < 0.75:
        sim.context.addSpike(snn.neurons[1])
    if t < 700 and random.random() < 0.25:
        sim.context.addSpike(snn.neurons[2])
    sim.nextTick()
# print("Completed simulation")

fig = plt.figure()
ax = mplot3d.Axes3D(fig)

visNeurons = []
i = -1
for n in snn.neurons:
    visNeurons.append(ax.scatter(
        n.position[0],
        n.position[1],
        n.position[2], c='#00ee00'))
for n in snn.neurons:
    for s in n.outputs:
        color = "red"
        if s.weight < 0:
            color = "blue"
        ax.plot([n.position[0], s.target.position[0]], [n.position[1], s.target.position[1]],
                zs=[n.position[2], s.target.position[2]], color=color, alpha=abs(s.weight) * 0.25)


def animate(frame):
    global visNeurons
    global sim
    global snn
    if random.random() < 0.75:
        sim.context.addSpike(snn.neurons[0])
    if random.random() < 0.75:
        sim.context.addSpike(snn.neurons[1])
    if random.random() < 0.25:
        sim.context.addSpike(snn.neurons[2])
    i = -1
    for n in snn.neurons:
        i += 1
        if n in sim.context.spikes:
            visNeurons[i].set_alpha(1)
        else:
            visNeurons[i].set_alpha(0.125)
    sim.nextTick()
    print("Simulation time: " + str(sim.context.clock))



ani = animation.FuncAnimation(fig, animate,
                              frames=25, interval=100, repeat=True)

# plotNeuronTracker(snn.neurons[0].membrane.tracker, "Neuron 1")
# plotNeuronTracker(snn.neurons[1].membrane.tracker, "Neuron 2")
# plotNeuronTracker(snn.neurons[2].membrane.tracker, "Neuron 3")
# plotNeuronTracker(snn.neurons[3].membrane.tracker, "Neuron 4")
show()
