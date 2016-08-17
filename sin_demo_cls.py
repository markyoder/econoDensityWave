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

class Sin_demo(object):
	def __init__(self, fignum=0):
		self.fig = plt.figure(fignum)
		self.fig.clf()
		#self.ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
		self.ax = plt.gca()
		#ax=self.ax
		self.ax.set_xlim(0.,2.)
		self.ax.set_ylim(-2.,2.)
		##
		n_points=1000
		#
		self.line, = self.ax.plot([], [], lw=2)	
		self.__dict__.update(locals())
		plt.show()

	def run(self):
		self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_plotter, frames=200, interval=20, blit=True)
		#
		# write all local variables to class scope:
		#self.__dict__.update(locals())
		

	# initialization function: plot the background of each frame
	def init_plotter(self):
		self.line.set_data([], [])
		return self.line,
	
	# animation function.  This is called sequentially
	def animate(self, i):
		x = np.linspace(0, 2, self.n_points)
		#
		y = np.sin(2 * np.pi * (x - 0.01 * i))
		#
		self.line.set_data(x, y)
		return self.line,
	#
	'''
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
	'''
	
if __name__=='__main__':
	#do_it()
	pass
else:
	plt.ion()
