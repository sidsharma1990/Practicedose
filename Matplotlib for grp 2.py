# Matplotlib (Visualization)

import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5,6]
y = [54,23,12,48,46,43]
plt.plot(x, y)

x = np.linspace(1,50,20)
y = np.random.randint(1,50,20)
y = np.sort(y)

plt.plot(x, y, 'r')
plt.plot(x, y, color = 'g')
plt.plot(x, y, color = 'black')

# Always run all codes together
plt.plot(x, y, color = 'g')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Random Graph')
plt.show()   # to remove text output


####### Subplot
# (1,2,1) = rows, column and graph number

plt.plot()

plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.subplot(1,2,3) # it wont work

plt.subplot(2,3,1)
plt.plot(x,y,'r')
plt.subplot(2,3,2)
plt.plot(x,y,'g')
plt.subplot(2,3,3)
plt.plot(x,y,'b')
plt.subplot(2,3,4)
plt.plot(x,y,'m')
plt.subplot(2,3,5)
plt.plot(x,y,'c')
plt.subplot(2,3,6)
plt.plot(x,y,'b')
plt.tight_layout()

##### Markers, line style
plt.subplot(1,2,1)
plt.plot(x,y,'r--')
plt.plot(x,y,'r-')
plt.plot(x,y,'r-.')
plt.plot(x,y,'r-')


plt.plot(x,y,'r+-')
plt.plot(x,y,'rv-')

plt.plot(x,y, linestyle = 'steps')
plt.plot(x,y, linestyle = '--', marker = 'P', color = 'black')


plt.plot(x,y, linestyle = '--', marker = 'P', 
         color = 'black', markersize = 7)


plt.plot(x,y, linestyle = '--', marker = 'P', 
         color = 'black', markersize = 15, 
         markerfacecolor = 'red')

plt.plot(x,y, linestyle = '--', marker = 'P', 
         color = 'black', markersize = 25, 
         markerfacecolor = 'red', markeredgecolor = 'cyan')

plt.plot(x,y, linestyle = '--', marker = 'P', 
         color = 'black', markersize = 25, 
         markerfacecolor = 'red', markeredgecolor = 'cyan',
         markeredgewidth = 5)

################################
plt.plot(x,y)
plt.scatter(x,y)
plt.hist(x,y)
plt.hist2d(x, y, bins = 10)
plt.bar(x,y)
plt.boxplot(x,y)
plt.polar(x,y)

# Heatmap
# num = np.random.random((4,4))
# plt.imshow(num, cmap = 'heat', interpolation = 'nearest')
# plt.show()

##### OOP (object oreiented plot)
x = np.linspace(1,50,20)
y = np.random.randint(1,50,20)
y = np.sort(y)

fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes1 = fig.add_axes([0.4,0.2,0.4,0.7])
axes.set_xlabel('Master X-Axis')
axes.set_ylabel('Master y-Axis')
axes.set_title('Master Plot')
axes1.set_xlabel('X-Axis')
axes1.set_ylabel('y-Axis')
axes1.set_title('Plot')
















