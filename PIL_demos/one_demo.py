import matplotlib.pyplot as plt
import numpy as np

# creating an empty object 
a= plt.figure(dpi=100, facecolor="#000000")

axes= a.add_axes([0.1,0.1,0.8,0.8])

# adding axes

x= np.arange(0,11)

axes.plot(x,x**3, marker='*')

axes.set_xlim([0,6])

axes.set_ylim([0,25])

plt.show()