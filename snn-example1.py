from numpy import *
from pylab import *
from model import *
import matplotlib.animation
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

snn = Snn.create(1500)

tmax = 1000
dt = 0.5
T = int(ceil(tmax / dt))

sim = Simulation([snn], dt)

# loop over time
for t in arange(T - 1):
    if t < 700 and random(1) < 0.75:
        snn.neurons[0].membrane.impulse(sim.context, 10)
    if t < 700 and random(1) < 0.75:
        snn.neurons[1].membrane.impulse(sim.context, 10)
    if t < 700 and random(1) < 0.25:
        snn.neurons[2].membrane.impulse(sim.context, 10)
    sim.nextTick()


fig = plt.figure()
ax = mplot3d.Axes3D(fig)
vis = ax.scatter(
    [n.position[0] for n in snn.neurons],
    [n.position[1] for n in snn.neurons],
    [n.position[2] for n in snn.neurons], c='#00ee00')

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

# plotNeuronTracker(snn.neurons[0].membrane.tracker, "Neuron 1")
# plotNeuronTracker(snn.neurons[1].membrane.tracker, "Neuron 2")
# plotNeuronTracker(snn.neurons[2].membrane.tracker, "Neuron 3")
# plotNeuronTracker(snn.neurons[3].membrane.tracker, "Neuron 4")
show()
