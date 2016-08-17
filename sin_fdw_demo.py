"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""



import numpy as np
numpy = np
from matplotlib import pyplot as plt
from matplotlib import animation

n_points=1000
B=10.
phi=0.

'''
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)	

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
plt.show()
'''
#plt.ion()
#
#
# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, n_points)
    #
    #f_active = lambda x: numpy
    #
    t = 2 * np.pi * (x - 0.01 * i)
    #y = np.sin(2 * np.pi * (x - 0.01 * i))
    #y = f_active(t)
    y = fdw(t, B=B, phi=phi)
    #
    line.set_data(x, y)
    return line,

def f_active(x):
	return numpy.sin(x)

def fdw(x=0,phi=0,B=10):
	A=1.
	#B=10.
	#phi=0.
	# nominally, phi=v*t, but ultimately is the phase of the cos part of the system.
	# A always =1; effectively, B -> B/A.
	return A*x**2 + B*numpy.cos(x + phi)

def do_it():
	

	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate, init_func=init,
		                           frames=200, interval=20, blit=True)

	# save the animation as an mp4.  This requires ffmpeg or mencoder to be
	# installed.  The extra_args ensure that the x264 codec is used, so that
	# the video can be embedded in html5.  You may need to adjust this for
	# your system: for more information, see
	# http://matplotlib.sourceforge.net/api/animation_api.html
	anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

	plt.show()

if __name__=='__main__':
	#do_it()
	
	fig = plt.figure()
	ax = plt.axes(xlim=(-5, 5), ylim=(-(B+2), B+2))
	line, = ax.plot([], [], lw=2)	

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
	plt.show()
	
#

else:
	plt.ion()
