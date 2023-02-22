import holoocean
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import controlAUV as control

scenario = "OpenWater-KonbesgergImagingSonar"
simulationConfiguration = holoocean.packagemanager.get_scenario(scenario)

#### ECHOSOUNDER CONFIG
simConfigSonar = simulationConfiguration['agents'][0]['sensors'][-1]["configuration"]
azimuth = simConfigSonar['Azimuth']
minRange = simConfigSonar['RangeMin']
maxRange = simConfigSonar['RangeMax']
binsRange = simConfigSonar['RangeBins']
binsAzimuth = simConfigSonar['AzimuthBins']

#### GET PLOT READY

# Enable interactive mode.
plt.ion()

### generate polar representation for WCI
figure, axis = plt.subplots(subplot_kw=dict(polar=True))

# set the 0 to -pi/2 so for a better representation of the WCI
axis.set_theta_zero_location("N")

## limit the polar representation of data according to the echosounder Azimuth

# Set the minimum theta limit in degrees.
axis.set_thetamin(-azimuth / 2)

# Set the maximum theta limit in degrees.
axis.set_thetamax(azimuth / 2)

## pixel representation: (r, theta) -> range and theta
# evenly space possible theta values. The azimuth bins state how the angles are spaced (pixels)
# theta in radians
theta = np.linspace(-azimuth / 2, azimuth / 2, binsAzimuth) * np.pi / 180

# evenly space numbers over a specified interval. The azimuth bins state how the values are spaced.
sonarRange = np.linspace(minRange, maxRange, binsRange)

thetaMesh, rangeMesh = np.meshgrid(theta, sonarRange)

backscatterZeroes = np.zeros_like(thetaMesh)

plt.grid(False)

# paint the SONAR representation
plot = axis.pcolormesh(thetaMesh, rangeMesh, backscatterZeroes, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
figure.canvas.draw()
figure.canvas.flush_events()

with tf.device('/GPU:0'):
    with holoocean.make(scenario_cfg=simulationConfiguration) as env:
        env.spawn_prop("box", location=[2, 0, -15], rotation=None, scale=1, sim_physics=False, material="brick",
                       tag="box_1")
        while True:
            if 'q' in control.pressed_keys:
                break
            command = control.parse_keys(control.pressed_keys, control.force)
            env.act("auv0", command)
            state = env.tick()
            if 'ImagingSonar' in state:
                s = state['ImagingSonar']
                plot.set_array(s.ravel())
                figure.canvas.draw()
                figure.canvas.flush_events()

print("Finished Simulation!")
plt.ioff()
plt.show()
