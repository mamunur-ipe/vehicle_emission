
class Car:
    a = 0   # acceleration in meter per second^2
    v = 0   # velocity in meter per second
    
    def __init__(self, mass, odometer_reading = 'low'):
        self.mass = mass
        self.odometer_reading = odometer_reading
    
    def vsp(self):
        a = self.a
        v = self.v
        mass = self.mass
        return v*(a+0.0981)+(0.49*pow(v,3)/mass)
    
    def vsp_to_bin(self, vsp):
        if vsp < -2:
        	bin = 1
        elif vsp >= -2 and vsp < 0:
        	bin = 2
        elif vsp >= 0 and vsp < 1:
        	bin = 3
        elif vsp >= 1 and vsp < 4:
        	bin = 4	
        elif vsp >= 4 and vsp < 7:
        	bin = 5
        elif vsp >= 7 and vsp < 10:
        	bin = 6
        elif vsp >= 10 and vsp < 13:
        	bin = 7
        elif vsp >= 13 and vsp < 16:
        	bin = 8
        elif vsp >= 16 and vsp < 19:
        	bin = 9
        elif vsp >= 19 and vsp < 23:
        	bin = 10
        elif vsp >= 23 and vsp < 28:
        	bin = 11
        elif vsp >= 28 and vsp < 33:
        	bin = 12
        elif vsp >= 33 and vsp < 39:
        	bin = 13
        else:
        	bin = 14
        
        return bin

    
    def bin_to_co2(self, bin):
        odometer_reading = self.odometer_reading
        if odometer_reading == "low":
        	if bin == 1:
        		co2 = 1.671
        	elif bin == 2:
        		co2 = 1.458
        	elif bin == 3:
        		co2 = 1.135
        	elif bin == 4:
        		co2 = 2.233
        	elif bin == 5:
        		co2 = 2.92
        	elif bin == 6:
        		co2 = 3.525
        	elif bin == 7:
        		co2 = 4.107
        	elif bin == 8:
        		co2 = 4.635
        	elif bin == 9:
        		co2 = 5.161
        	elif bin == 10:
        		co2 = 5.633
        	elif bin == 11:
        		co2 = 6.535
        	elif bin == 12:
        		co2 = 7.585
        	elif bin == 13:
        		co2 = 9.024
        	else:
        		co2 = 10.088
        
        else:
        	if bin == 1:
        		co2 = 1.544
        	elif bin == 2:
        		co2 = 1.604
        	elif bin == 3:
        		co2 = 1.131
        	elif bin == 4:
        		co2 = 2.386
        	elif bin == 5:
        		co2 = 3.21
        	elif bin == 6:
        		co2 = 3.958
        	elif bin == 7:
        		co2 = 4.752
        	elif bin == 8:
        		co2 = 5.374
        	elif bin == 9:
        		co2 = 5.94
        	elif bin == 10:
        		co2 = 6.428
        	elif bin == 11:
        		co2 = 7.066
        	elif bin == 12:
        		co2 = 7.618
        	elif bin == 13:
        		co2 = 8.322
        	else:
        		co2 = 8.475

        return co2
 
# define a function which will convert speed mph to mps        
def mph_to_mps(mph):
    return mph*1.60934*1000/3600

# create a car object c
c = Car(mass = 1600, odometer_reading ='low')

# create speed and acceleration range for which analysis will be done
import numpy as np
#speed_mph = range(1, 81)     # in mile per hour
speed_mph = np.arange(5, 71, 5)
speeds = [mph_to_mps(x) for x in speed_mph]
accelerations = range(0, 5)

# create a pandas data frame where the output data will be stored
import pandas as pd
df = pd.DataFrame(columns = ['acceleration', 'speed_mph', 'vsp', 'bin', 'co2', 'co2_ratio'])
df.head()

counter = 0     # created for denoting the dataframe row number
# execute iteration and store the data
for acceleration in accelerations:
    for speed in speed_mph:
        c.v = mph_to_mps(speed)     # convert speed from mph to pms
        c.a = acceleration
        vsp = c.vsp()
        bin = c.vsp_to_bin(vsp)
        co2 = c.bin_to_co2(bin)
        df.loc[counter, 'speed_mph'] = speed
        df.loc[counter, 'acceleration'] = acceleration
        df.loc[counter, 'vsp'] = vsp
        df.loc[counter, 'bin'] = bin
        df.loc[counter, 'co2'] = co2
        df.loc[counter, 'co2_ratio'] = round(co2/speed, 3)
        
        counter += 1



## create 2D plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(constrained_layout= True, figsize=(6, 5))
gs = GridSpec(1, 1, figure=fig)

ax = fig.add_subplot(gs[0, 0])

unique = df['acceleration'].unique()
for i in unique:
    df_new = df[df['acceleration'] == unique[i]]
    x = df_new['speed_mph'].values
    y = df_new['co2_ratio'].values
#    ax.scatter(x, y)
#    ax.plot(x, y)
    ##make a smooth line plot
    from scipy.interpolate import make_interp_spline
    # 300 represents number of points to make between x.min and x.max
    x_new = np.linspace(x.min(), x.max(), 300) 
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)
    ax.plot(x_new, y_smooth)


# set legend
legend_text = [f"acceleration = {x}" for x in unique]
legend = ax.legend(legend_text, loc='upper right', fontsize=10)
legend.get_frame().set_edgecolor('k')
#set axis labels
ax.set_xlabel('Speed (mph)', fontsize=11, color='b')
ax.set_ylabel('CO\N{SUBSCRIPT TWO}/speed ratio (g/mile)', fontsize=11, color='b')
#ax.set_yticks(np.arange(0, 41, 10))
plt.show()
#fig.savefig("emission_plot_2D.png", dpi=300, bbox_inches='tight')


## create a 3D surface plot
#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(constrained_layout= True, figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

x = df['speed_mph'].values
y = df['acceleration'].values
z = df['co2_ratio'].values
from scipy.interpolate import griddata
X, Y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
Z = griddata((x, y), z, (X, Y), method='cubic')

#ax.plot_surface(X, Y, Z)
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
#cmap = 'YlGn'
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=15)
#ax.plot_wireframe(X, Y, Z)
plt.show()
#fig.savefig("emission_plot_3D.png", dpi=300, bbox_inches='tight')



