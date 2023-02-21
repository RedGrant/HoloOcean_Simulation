import holoocean
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

scenario = "OpenWater-KonbesgergImagingSonar"
simulationConfiguration = holoocean.packagemanager.get_scenario(scenario)



#### GET SONAR CONFIG
simConfigSonar = simulationConfiguration['agents'][0]['sensors'][-1]["configuration"]
azimuth = simConfigSonar['Azimuth']
minRange = simConfigSonar['RangeMin']
maxRange = simConfigSonar['RangeMax']
binsRange = simConfigSonar['RangeBins']
binsAzimuth = simConfigSonar['AzimuthBins']

command = np.array([0, 0, 0, 0, -20, -20, -20, -20])
plt.grid(False)


with tf.device('/GPU:0'):
    with holoocean.make(scenario_cfg= simulationConfiguration) as env:
        env.spawn_prop("box", location=[0, 0, -2], rotation=None, scale=1, sim_physics=False, material="wood",
                       tag="box_1")
        for _ in range(1000):
            env.act("auv0", command)
            state = env.tick()
            if 'ImagingSonar' in state:
                s = state['ImagingSonar']







#### GET PLOT READY
#plt.ion()
#fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
#ax.set_theta_zero_location("N")
#ax.set_thetamin(-azi / 2)
#ax.set_thetamax(azi / 2)
#
#theta = np.linspace(-azi / 2, azi / 2, binsA) * np.pi / 180
#r = np.linspace(minR, maxR, binsR)
#T, R = np.meshgrid(theta, r)
#z = np.zeros_like(T)
#
#plt.grid(False)
#plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
#plt.tight_layout()
#fig.canvas.draw()
#fig.canvas.flush_events()
#
##### RUN SIMULATION
#command = np.array([0, 0, 0, 0, -20, -20, -20, -20])
#
#with tf.device('/GPU:0'):
#    with holoocean.make(scenario) as env:
#        for i in range(1000):
#            env.act("auv0", command)
#            state = env.tick()
#
#            if 'ImagingSonar' in state:
#                s = state['ImagingSonar']
#                plot.set_array(s.ravel())
#
#                fig.canvas.draw()
#                fig.canvas.flush_events()
#
#    print("Finished Simulation!")
#    plt.ioff()
#    plt.show()
#