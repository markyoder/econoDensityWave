'''
#econo-DensityWave:
# density wave simulations.
# This needs some tuning. there are two working scripts at the end that will produce a bunch of still figs and maybe even a movie.
# This is supposed to produce a real-time movie in the matplotlib qt window -- and i think it worked under earlier versions of
# Python 2.x, but does not seem to be working in Anaconda Python 3.x.
#
# for images, see two scripts at the end of this module; see the syntax for the mencoder script as well. note also that
# mencoder seems to be a library a bit in flux, so there might be a bit more Google in the future of this as well.
'''
#
import matplotlib.pyplot as plt
#import scipy
import numpy
import math
import os
import sys

from matplotlib import animation

if sys.version_info.major==3: xrange=range

class Plotter(object):
	plt.ion()

	#def __init__(self, start, stop, beta, bigev=[]):
	def __init__(self, B=1, fignum=1, range=None, nlines=1, figsize=(5,4), legend_font_size=10 ):
		#self.start = start
		#self.stop  = stop

		self.fig = plt.figure(fignum, figsize=figsize)
		self.fig.clf()
		self.ax  = self.fig.add_subplot(111)
		self.ax.set_title('B/A = %s, time=0' % B)
		#self.line1 = self.ax.plot( [], [], '-r', label='fdw')[0]
		#self.line2 = self.ax.plot( [], [], 'ob', label='ball')[0]
		self.lines=[]
		#for i in range(nlines):
			#self.lines+=[self.ax.plot( [], [], '-r', label='fdw')[0]]
		self.ax.legend(loc='upper center', numpoints=1, fontsize=legend_font_size)
		

		#for event in bigev:
		#    if (self.start <= event <= self.stop):
		#        self.ax.axvline(x=pylab.date2num(event), color='k')

	def __del__(self):
		plt.show()

	def updateData2(self, X, Ys=[], Ysprams=[], B=0, phi=0, yrange=[]):
		if Ys==None: Ys=[]
		if type(Ys).__name__!='list' and type(Ys).__name__!='tuple': Ys=[]
		if yrange==None or yrange==[] or len(yrange)<2: Yrange=[-B, fdw(B, X[0], 0)]
		#	
		self.ax.set_title('$B/A = %f$, $time=%f$' % (B, phi))
		# are all the line objects represented?
		while len(Ysprams)<len(Ys):
			Ysprams+=[['-', '']]	# line format, label
		while len(self.lines)<len(Ys):
			newindex=len(self.lines)
			self.lines+=self.ax.plot( [], [], Ysprams[newindex][0], label=Ysprams[newindex][1])
			#self.ax.legend(loc='upper center', numpoints=1)
		#
		for i in range(len(self.lines)):
			self.lines[i].set_data(X[i], Ys[i])
			
		#
		self.ax.relim()
		self.ax.autoscale_view()
		self.ax.set_xlim([X[0][0], X[0][-1]])
		#self.ax.set_ylim([-10, 80])
		self.ax.set_ylim(yrange)
		self.fig.canvas.draw()
	
	def updateData(self, X, Y1=None, Y2=None, B=0, phi=0):
		if Y1==None: Y1=[]
		if Y2==None: Y2=[]
		#
		self.ax.set_title('Econo-Density wave, $B/A = %f$, time=%f' % (B, phi))
		self.line1.set_data(X, Y1)
		#self.line2.set_data(X, Y2)
		self.ax.relim()
		self.ax.autoscale_view()
		self.ax.set_xlim([X[0], X[-1]])
		self.ax.set_ylim([-10, 80])
		self.fig.canvas.draw()
	
	def setCanvas(self, xlims=[-1,1], ylims=[-1,1]):
		self.ax.relim()
		self.ax.autoscale_view()
		self.ax.set_xlim(xlims)
		self.ax.set_ylim(ylims)
		self.fig.canvas.draw()
	
	def updateLineData(self, X=None, Y=None, linenum=0, yRange=None, doDraw=True):
		#
		if yRange==None or yRange==[] or len(yRange)<2: yRange=self.getMaxMin(Y)
		#
		#self.ax.set_title('Econo-Density wave, $B/A = %f$, time=%f' % (B, phi))
		#self.line1.set_data(X, Y1)
		#self.line2.set_data(X, Y2)
		self.lines[linenum].set_data(X,Y)
		self.ax.relim()
		self.ax.autoscale_view()
		#self.ax.set_xlim([X[0], X[-1]])
		#self.ax.set_ylim([-10, 80])
		if yRange!=None: self.ax.set_ylim(yRange)
		if doDraw: self.fig.canvas.draw()
	
	def getMaxMin(self, data0=[], cols=[0,1]):
		# data: 2D list. use map(operator.itemgetter(n), list) if necessary.
		# cols are indices for [x, f(x)]
		#
		if type(data0[0]).__name__ in ['list', 'tuple']:
			print('list or tuple')
			data=data0	
		if type(data0[0]).__name__ in ['float', 'float64', 'float32'] or type(data0[0]).__name__ == 'int':
			# mock up a 2xN array with indices as x vals:
			#
			data=[]
			for i in range(len(data0)):
				data+=[[i, data0[i]]]
			#
		#
		#
		maxmin=[[data[0][0], data[0][1]], [data[0][0], data[0][1]]]
		for rw in data:
			if rw[1]<maxmin[0][1]: maxmin[0]=rw	# min.
			if rw[1]>maxmin[1][1]: maxmin[1]=rw	# max
		# (alternatively, do sort(), then X[0], X[-1], but we have to copy the array (to avoid by-reff problems), so this is probably equivalent.
		return maxmin
	#
	def rollBall(self, ballxIndex=0, Yvals=None, lright=None):
		# lright: only left or right? we'll probably never use this feature.
		# use this function to move the ball around.
		# look left/right (eventually, i guess we have to choose one randomly, for now keep it simple and go left first...
		# if y(x+/-1)<y0, go that way until we get to the bottom.
		# use "natural x" space (aka, use x indeces, not x values).
		#
		i=ballxIndex
		# go left:
		while i>1 and Yvals[i-1]<Yvals[i]:
			i-=1
		while i<(len(Yvals)-1) and Yvals[i+1]<Yvals[i]:
			i+=1
		return i

def fdw(B=10,x=0,phi=0):
	A=1
	# nominally, phi=v*t, but ultimately is the phase of the cos part of the system.
	# A always =1; effectively, B -> B/A.
	return A*x**2 + B*math.cos(x + phi)

def fdwlist(B=10., X=None, phi=0):
	#if type(X[0]).__name__=='int':
	if isinstance(X[0],int):
		X = [float(x) for x in X]
		#for i in range(len(X)):
		#	X[i]=float(X[i])
	#
	Y=[]
	minY=None
	for x in X:
		Y+=[fdw(B, x, phi)]
		if minY is None:
			minY=[x, Y[0]]
		elif Y[-1]<minY[1]:
			minY=[x, Y[-1]]
	return [Y, minY]

def dwaveFrames(B=10., phi=0., ballpos=None, fignum=0, Yrange=None):
	# just custom script this one.
	# make some dwave frames for john:
	# set up some constants, objects, and working values:
	#
	# also, movie scripts (so far) don't use this script; only "stills" does. if we return the lines array,
	# can we use it with an animator?
	#
	fignum=fignum
	#
	if Yrange==None: Yrange=[-(B+2), 80]
	ballPixSize=2.1
	myplot=Plotter(B, fignum)
	#phi=0.
	X=numpy.array(range(1000))
	x0=-8
	xmax=8
	fX=X*(xmax-x0)/float(len(X)-1)+x0	# array of float x values
	#
	#cyc_time_marker='^'
	cyc_time_marker=''
	#cyc_time_label='Cycle time'
	cyc_time_label = ''
	#
	myplot.setCanvas([fX[0], fX[-1]], Yrange)
	myplot.lines+=myplot.ax.plot([], [], '-b', lw=2)
	myplot.lines+=myplot.ax.plot([], [], color='r', marker=cyc_time_marker, label=cyc_time_label, ms=10)
	myplot.lines+=myplot.ax.plot([], [], 'go', ms=15)
	#
	lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
	Y=lY[0]
	minY=lY[1]	# actually [x,Ymin]
	#
	Yphi=[0]
	Xphi=[phi]
	#
	# ball position:
	if ballpos=='None' or ballpos=='min':
		# we've already gathered these values.
		ballx=[minY[0]]
		bally=[minY[1]+ballPixSize]
	elif ballpos=='left':
		#getMaxMin(self, data0=[], cols=[0,1]):
		mm=myplot.getMaxMin(Y[0:int(len(Y)/2)])
		ballx=[fX[mm[0][0]]]	# we pass a 1D array, so we get back [index, yval]
		bally=[mm[0][1]+ballPixSize]
		#
	elif ballpos=='right':
		#getMaxMin(self, data0=[], cols=[0,1]):
		mm=myplot.getMaxMin(Y[-int(len(Y)/2):])
		ballx=[fX[mm[0][0] + int(len(Y)/2)]]	# we pass a 1D array, so we get back [index, yval]
		bally=[mm[0][1]+ballPixSize]
		#
	elif type(ballpos).__name__=='list':
		# specify the range...
		inds=[0, len(fX)-1]	# indeces range
		for i in range(len(fX)):
			x=fX[i]
			if x<ballpos[0]: inds[0]=i
			if x<ballpos[1]: inds[1]=i
			if x>ballpos[1]: break
		mm=myplot.getMaxMin(Y[inds[0]:inds[1]])
		ballx=[fX[mm[0][0]+inds[0]]]	# we pass a 1D array, so we get back [index, yval]
		bally=[mm[0][1]+ballPixSize]
	else:
		ballx=[minY[0]]
		bally=[minY[1]+ballPixSize]
	#
	
	# meta-unstable?
	if (bally[0]-ballPixSize)>(minY[1]):
		if myplot.lines[2].get_mfc()!='r': myplot.lines[2].set_markerfacecolor('r')
	else:
		if myplot.lines[2].get_mfc()!='g': myplot.lines[2].set_mfc('g')		#
	#
	myplot.updateLineData(fX, Y, 0, Yrange, False)
	myplot.updateLineData([phi], [-11], 1, Yrange, False)
	myplot.updateLineData(ballx, bally, 2, Yrange, True)
	myplot.ax.set_title('$B/A = %f$, $time=%f$' % (B, phi))
	
	return fX

#
# development and practice bits:

def doDwaveMovie1(B=10, fignum=0, dosave=False):
	#fignum=0
	#plt.figure(fignum)
	#plt.ion()
	myplot=Plotter(B, fignum)
	#
	# set up:
	phiRes=100
	#cyc_time_marker='^'
	cyc_time_marker=''
	#cyc_time_label = 'Cycle time'
	cyc_time_label = ''
	
	X=numpy.array(range(1001))	# 100 steps per period/cycle.
	# because of the quadratic term, the function is not periodic, so we are not limited to dX=2pi. +/-2pi is pretty nice, +/-10 or so is nicely illustrative.
	x0=-8.
	xmax=8.
	fX=X*(xmax-x0)/float(len(X)-1)+x0
	#
	phi=fX[0]	# assume we run phi through the full range of fX
	iframe=0
	indexString0=''
	indexlen=2+int(math.log10(4+phiRes*(1+B/2)))
	for i in range(indexlen):
		indexString0+='0'
	#	
	#Yprams=[['-b', 'FDW(x,t)'], ['r{}'.format(cyc_time_marker), 'time'], ['gv', '']]
	Yprams=[['-b', 'FDW(x,t)'], ['r{}'.format(cyc_time_marker), 'time'], ['go', '']]
	# and we know we want 3 lines, so let's define them here:
	# self.lines+=self.ax.plot( [], [], Ysprams[newindex][0], label=Ysprams[newindex][1])
	
	#
	# once through:
	while phi<fX[-1]:
		lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
		Y=lY[0]
		minY=lY[1]	# actually [x,Ymin]
		#
		Yphi=[0]
		Xphi=[phi]
		#
		#myplot.updateData(fX, Y, [], B, phi)
		# updateData2(self, X, Ys=[], Ysprams=[], B=0, phi=0, yrange=[], dosave=False)
		myplot.updateData2([fX, Xphi, [minY[0]] ], [Y, Yphi, [minY[1]+2] ], Yprams, B, phi, [-12, 80])
		plt.draw()
		#
		if dosave:
		#if not foutname is None:
			indexString="%s%s" % (indexString0, str(iframe))
			indexString=indexString[-indexlen:]
			fnamebase='images1/dwaveFrame'
			fname='%s-%s.png' % (fnamebase, indexString)
			fnamejpg='%s-%s.jpg' % (fnamebase, indexString)
			myplot.fig.savefig(fname)
			os.system('convert %s %s' % (fname, fnamejpg))
		
		#
		phi+=(xmax-x0)/float(phiRes)
		
		iframe+=1
	#
	# and for other B values:
	#phiRes=25
	while B>0:
		phi=fX[0]
		while phi<fX[-1]:
			lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
			Y=lY[0]
			minY=lY[1]	# actually [x,Ymin]
		#
			Yphi=[0]
			Xphi=[phi]
			#myplot.updateData2(fX, Y, [], B, phi)
			myplot.updateData2([fX, Xphi, [minY[0]] ], [Y, Yphi, [minY[1]+2]], Yprams, B, phi, [-12, 80])
			#
			if dosave:
			#if not foutname is None:
				indexString="%s%s" % (indexString0, str(iframe))
				indexString=indexString[-indexlen:]
				fnamebase='images1/dwaveFrame'
				fname='%s-%s.png' % (fnamebase, indexString)
				fnamejpg='%s-%s.jpg' % (fnamebase, indexString)
				myplot.fig.savefig(fname)
				os.system('convert %s %s' % (fname, fnamejpg))
			#
			phi+=(xmax-x0)/float(phiRes)
		
			iframe+=1
		if B>2:
			B-=2
		else:
			B-=.5
		
	if dosave:
	#if not foutname is None:
		#os.system('mencoder mf:///home/myoder/Documents/Research/Rundle/econoPhysics/econoDensityWave/images1/*.jpg -mf w=800:h=600:fps=10:type=jpg -ovc copy -o econoDwaveMovie.avi')
		f_path = os.path.join(os.getcwd(), 'images_b', '*.jpg')
		os.system(os.system('mencoder mf://{} -mf w=800:h=600:fps=10:type=jpg -ovc copy -o econoDwaveMovie.avi'.format(f_path)))
	return None


def doDwaveMovie2(B=10, dosave=False):
	fignum=0
	#plt.figure(fignum)
	#plt.ion()
	myplot=Plotter(B, fignum)
	#
	# set up:
	phiRes=100
	#cyc_time_marker='^'
	cyc_time_marker=''
	#cyc_time_label = 'Cycle time'
	cyc_time_label = ''
	
	X=numpy.array(range(1001))	# 100 steps per period/cycle.
	# because of the quadratic term, the function is not periodic, so we are not limited to dX=2pi. +/-2pi is pretty nice, +/-10 or so is nicely illustrative.
	x0=-8.
	xmax=8.
	fX=X*(xmax-x0)/float(len(X)-1)+x0
	#
	phi=fX[0]	# assume we run phi through the full range of fX
	iframe=0
	indexString0=''
	indexlen=2+int(math.log10(4+phiRes*(1+B/2)))
	for i in range(indexlen):
		indexString0+='0'
	#	
	#Yprams=[['-b', 'FDW(x,t)'], ['r{}'.format(cyc_time_marker), 'time'], ['gv', '']]
	Yprams=[['-b', 'FDW(x,t)'], ['r{}'.format(cyc_time_marker), 'time'], ['go', '']]
	# and we know we want 3 lines, so let's define them here:
	# self.lines+=self.ax.plot( [], [], Ysprams[newindex][0], label=Ysprams[newindex][1])
	myplot.setCanvas([fX[0], fX[-1]], [-B-2, 80])
	myplot.lines+=myplot.ax.plot([], [], '-', lw=2, color='b')
	myplot.lines+=myplot.ax.plot([], [], 'r{}'.format(cyc_time_marker), label='time', ms=10)
	myplot.lines+=myplot.ax.plot([], [], 'o', color='g', ms=15)
	myplot.ax.legend(loc='best', numpoints=1)
	#
	Yrange=[-(B+2), 80]
	ballPixSize=2.1
	#
	# once through:
	while phi<fX[-1]:
		lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
		Y=lY[0]
		minY=lY[1]	# actually [x,Ymin]
		#
		Yphi=[0]
		Xphi=[phi]
		#
		#myplot.updateData(fX, Y, [], B, phi)
		# updateData2(self, X, Ys=[], Ysprams=[], B=0, phi=0, yrange=[], dosave=False)
		# myplot.updateData2([fX, Xphi, [minY[0]] ], [Y, Yphi, [minY[1]+2] ], Yprams, B, phi, [-12, 80])
		# updateLineData(self, X, Y, linenum=0, yRange=[], doDraw=True)
		myplot.updateLineData(fX, Y, 0, Yrange, False)
		myplot.updateLineData([phi], [-B], 1, Yrange, False)
		myplot.updateLineData([minY[0]], [minY[1]+ballPixSize], 2, Yrange, True)
		myplot.ax.set_title('Econo-Density wave, $B/A = %f$, $time=%f$' % (B, phi))
		#
		if dosave:
			indexString="%s%s" % (indexString0, str(iframe))
			indexString=indexString[-indexlen:]
			fnamebase='images1/dwaveFrame'
			fname='%s-%s.png' % (fnamebase, indexString)
			fnamejpg='%s-%s.jpg' % (fnamebase, indexString)
			myplot.fig.savefig(fname)
			os.system('convert %s %s' % (fname, fnamejpg))
		
		#
		phi+=(xmax-x0)/float(phiRes)
		
		iframe+=1
	#
	# and for other B values:
	#phiRes=25
	while B>0:
		phi=fX[0]
		while phi<fX[-1]:
			lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
			Y=lY[0]
			minY=lY[1]	# actually [x,Ymin]
		#
			Yphi=[0]
			Xphi=[phi]
			#myplot.updateData2(fX, Y, [], B, phi)
			#myplot.updateData2([fX, Xphi, [minY[0]] ], [Y, Yphi, [minY[1]+2]], Yprams, B, phi, [-12, 80])
			myplot.updateLineData(fX, Y, 0, Yrange, False)
			myplot.updateLineData([phi], [-B], 1, Yrange, False)
			myplot.updateLineData([minY[0]], [minY[1]+ballPixSize], 2, Yrange, True)
			myplot.ax.set_title('Econo-Density wave, $B/A = %f$, $time=%f$' % (B, phi))
			#
			if dosave:
				indexString="%s%s" % (indexString0, str(iframe))
				indexString=indexString[-indexlen:]
				fnamebase='images1/dwaveFrame'
				fname='%s-%s.png' % (fnamebase, indexString)
				fnamejpg='%s-%s.jpg' % (fnamebase, indexString)
				myplot.fig.savefig(fname)
				os.system('convert %s %s' % (fname, fnamejpg))
			#
			phi+=(xmax-x0)/float(phiRes)
		
			iframe+=1
		if B>2:
			B-=2
		else:
			B-=.5
		
	#if dosave: os.system('mencoder mf:///home/myoder/Documents/Research/Rundle/econoPhysics/econoDensityWave/images1/*.jpg -mf w=800:h=600:fps=10:type=jpg -ovc copy -o econoDwaveMovie.avi')
	#
	if dosave:
		pth = os.path.join(os.getcwd(), 'images1')
		pth = os.path.join(pth, '*.jpg')
		os.system('mencoder mf://{} -mf w=800:h=600:fps=10:type=jpg -ovc copy -o econoDwaveMovie.avi'.format(pth))
	return None
	
	
def practiceBits(B=10, phi0=0):
	plt.figure(0)
	plt.clf()
	
	#plt.ion()
	#
	# set up:
	X=numpy.array(range(101))	# 100 steps per period/cycle.
	# because of the quadratic term, the function is not periodic, so we are not limited to dX=2pi. +/-2pi is pretty nice, +/-10 or so is nicely illustrative.
	x0=-8.
	xmax=8.
	fX=X*(xmax-x0)/float(len(X)-1)+x0
	#
	Y0=fdwlist(B, fX, 0)
	Y1=fdwlist(B, fX, 1.0)
	Y2=fdwlist(B, fX, 2.0)
	Y3=fdwlist(B, fX, 3.0)
	#
	plt.plot(fX, Y0)
	plt.plot(fX, Y1)
	plt.plot(fX, Y2)
	plt.plot(fX, Y3)
	#
	
####################################################
####################################################
#
def doDwaveMovie(B0=10, dosave=False, imagesdir='images1', lbls=['Stable asset value', 'Metastable asset value'], movieName='dwavemovie1.avi',x_label='Deficit', y_label='Potential', figsize=(4,3), legend_font_size=10, save_jpeg=False):
	# production script:
	# this just like movie2, but use the "find minY" algorithm to move the ball around.
	#
	# yoder: but get rid of the changinf B value (JBR was right...).
	#
	if not os.path.isdir(imagesdir): os.makedirs(imagesdir)
	fignum=0
	B=B0
	#plt.figure(fignum)
	#plt.ion()
	myplot=Plotter(B, fignum, figsize=figsize, legend_font_size=legend_font_size)
	#
	# set up:
	phiRes = 50            # "phi resolution". i think what i've done here is, in a convoluted way, set this up so that X has much
	#                      # higher resolution than we actually use, mainly for evaluating whether the ball is in a basin or not (i think).
	#                      # problem is, i think this gives us some odd behaviors when the ball is near the bottom of a basin, but not quite
	#                      # there; it looks meta-stable... which we can call a fluctuation if we want...
	cyc_time_marker='^'
	cyc_time_marker=''
	#cyc_time_label = 'Cycle time'
	cyc_time_label = ''
	cyc_time_color = 'r'
	
	#X=numpy.array(range(10001))	# steps per period/cycle.
	# because of the quadratic term, the function is not periodic, so we are not limited to dX=2pi. +/-2pi is pretty nice, +/-10 or so is nicely illustrative.
	x0=-8.
	# original xmax:
	#xmax=8.
	# working xmax:
	xmax=6
	#
	stable_ball_color='b'
	meta_ball_color='r'
	ball_scatter_size = 100
	ball_lw = 2.5
	#
	#fX=X*((xmax-x0)/float(len(X)-1))+x0		# or next time, just use numpy.linspace()
	fX = numpy.linspace(x0, xmax, 10000)
	#
	phi=fX[0]	# assume we run phi through the full range of fX
	iframe=0
	indexString0=''
	indexlen=2+int(math.log10(4+phiRes*(1+B/2)))
	for i in range(indexlen):
		indexString0+='0'
	#	
	#Yprams=[['-b', 'FDW(x,t)'], ['r{}'.format(cyc_time_marker), 'time'], ['gv', '']]
	Yprams=[['-b', 'FDW(x,t)'], ['r{}'.format(cyc_time_marker), 'time'], ['go', '']]
	# and we know we want 3 lines, so let's define them here:
	# self.lines+=self.ax.plot( [], [], Ysprams[newindex][0], label=Ysprams[newindex][1])
	myplot.setCanvas([fX[0], fX[-1]], [-B-2, 80])
	myplot.lines+=myplot.ax.plot([], [], '-', lw=2, color='b')
	myplot.lines+=myplot.ax.plot([], [], color=cyc_time_color, marker=cyc_time_marker, label=cyc_time_label, ms=10)
	#
	myplot.lines+=myplot.ax.plot([], [], 'o', color=stable_ball_color, label=lbls[0], ms=15, zorder=2)
	myplot.lines+=myplot.ax.plot([], [], 'o', color=meta_ball_color, label=lbls[1], ms=15, zorder=5)
	#
	# this is how to draw the points using scatter(), and we can have hollow circles. BUT, this does not add to the plt.lines[] object,
	# so we'll have to rewrite the code that updates them... later.
	#myplot.lines+=myplot.ax.scatter([], [], marker='o', s=ball_scatter_size, lw=ball_lw, color=stable_ball_color, label=lbls[0], ms=15)
	#myplot.lines+=myplot.ax.scatter([], [], marker='o', s=ball_scatter_size, lw=ball_lw, color=meta_ball_color,   label=lbls[1], ms=15)
	#
	myplot.ax.legend(loc='upper right', numpoints=1)
	#
	Yrange=[-(B+2), 80]
	ballPixSize=2.1 
	ballIndex=int(len(fX)/2)	# start it in the middle...
	#
	# once through:
	k=0
	while phi<fX[-1]:
		k+=1
		lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
		Y=lY[0]
		minY=lY[1]	# actually [x,Ymin]
		#
		# and using ball rolling algorithm:
		#def rollBall(self, ballxIndex=0, Yvals, lright=None):
		ballIndex=myplot.rollBall(ballIndex, Y)
		ballpos=[fX[ballIndex], Y[ballIndex]+ballPixSize]
		#
		# meta-unstable?
		# meta-unstable?
		#if k<10: print('ball color data: ', k, ballpos[1],min(Y),  (ballpos[1]-ballPixSize)>(minY[1]),  ballPixSize)
		if (ballpos[1]-ballPixSize)>(minY[1]):
		#if (ballpos[1]>min(Y)):
			#if k<10: print('setting color: ', meta_ball_color)
			#myplot.lines[2].set_markerfacecolor(meta_ball_color)
			myplot.lines[2].set_markerfacecolor(meta_ball_color)
			#
			# are we seeing the other ball?
			#myplot.lines[3].set_markerfacecolor(meta_ball_color)
			#
			#if myplot.lines[2].get_mfc()!=meta_ball_color: myplot.lines[2].set_markerfacecolor(meta_ball_color)
			#if myplot.lines[2].get_mfc()!=meta_ball_color: myplot.lines[2].set_markerfacecolor(meta_ball_color)
		else:
			#if myplot.lines[2].get_mfc()!=stable_ball_color: myplot.lines[2].set_mfc(stable_ball_color)		#
			myplot.lines[2].set_markerfacecolor(stable_ball_color)		#
			#myplot.lines[3].set_markerfacecolor(stable_ball_color)		#
		#Yphi=[0]
		#Xphi=[phi]
		#
		#myplot.updateData(fX, Y, [], B, phi)
		# updateData2(self, X, Ys=[], Ysprams=[], B=0, phi=0, yrange=[], dosave=False)
		# myplot.updateData2([fX, Xphi, [minY[0]] ], [Y, Yphi, [minY[1]+2] ], Yprams, B, phi, [-12, 80])
		# updateLineData(self, X, Y, linenum=0, yRange=[], doDraw=True)
		myplot.updateLineData(fX, Y, 0, Yrange, False)
		myplot.updateLineData([phi], [-B0], 1, Yrange, False)
		#myplot.updateLineData([minY[0]], [minY[1]+ballPixSize], 2, Yrange, True)		# draw "ball" at min. location
		
		#print('***DEBUG: ', [ballpos[0]], [ballpos[1]], 2, Yrange, True)
		myplot.updateLineData([ballpos[0]], [ballpos[1]], 2, Yrange, True)
		#myplot.updateLineData([ballpos[0]], [ballpos[1]], 3, Yrange, True)
		myplot.ax.set_title('$B/A = %f$, $time=%f$' % (B, phi))
		#
		myplot.ax.set_xlabel(x_label)
		myplot.ax.set_ylabel(y_label)
		#
		if dosave:
			indexString="%s%s" % (indexString0, str(iframe))
			indexString=indexString[-indexlen:]
			fnamebase='%s/dwaveFrame' % imagesdir
			fname='%s-%s.png' % (fnamebase, indexString)
			myplot.fig.savefig(fname)
			if save_jpeg:
				fnamejpg='%s-%s.jpg' % (fnamebase, indexString)
				os.system('convert %s %s' % (fname, fnamejpg))
		
		#
		phi+=(xmax-x0)/float(phiRes)
		
		iframe+=1
	#
	# and for other B values:
	#phiRes=25
	# yoder:
	#while B>0:
	while False:
		phi=fX[0]
		while phi<fX[-1]:
			lY=fdwlist(B, fX, phi)	# list returned by fdwlist(): [Yvals{list}, Ymin{list: [x,y]}]
			Y=lY[0]
			minY=lY[1]	# actually [x,Ymin]
			x_min_Y, min_Y = lY[1]
			#
			ballIndex=myplot.rollBall(ballIndex, Y)
			ballpos=[fX[ballIndex], Y[ballIndex]+ballPixSize]
			#
			# meta-unstable?
			#if (ballpos[1]-ballPixSize)>(minY[1]):
			if ballpos[1] > (min_Y):
				#if myplot.lines[2].get_mfc()!=meta_ball_color: myplot.lines[2].set_markerfacecolor(meta_ball_color)
				myplot.lines[2].set_markerfacecolor(meta_ball_color)
			else:
				#if myplot.lines[2].get_mfc()!=stable_ball_color: myplot.lines[2].set_markerfacecolor(stable_ball_color)
				myplot.lines[2].set_markerfacecolor(stable_ball_color)
			
		#
			#Yphi=[0]
			#Xphi=[phi]
			#myplot.updateData2(fX, Y, [], B, phi)
			#myplot.updateData2([fX, Xphi, [minY[0]] ], [Y, Yphi, [minY[1]+2]], Yprams, B, phi, [-12, 80])
			myplot.updateLineData(fX, Y, 0, Yrange, False)
			myplot.updateLineData([phi], [-B0], 1, Yrange, False)
			#myplot.updateLineData([minY[0]], [minY[1]+ballPixSize], 2, Yrange, True)
			myplot.updateLineData([ballpos[0]], [ballpos[1]], 2, Yrange, True)
			myplot.ax.set_title('$B/A = %f$, $time=%f$' % (B, phi))
			#
			if dosave:
				indexString="%s%s" % (indexString0, str(iframe))
				indexString=indexString[-indexlen:]
				fnamebase='%s/dwaveFrame' % imagesdir
				fname='%s-%s.png' % (fnamebase, indexString)
				myplot.fig.savefig(fname)
				if save_jpeg:
					fnamejpg='%s-%s.jpg' % (fnamebase, indexString)
					os.system('convert %s %s' % (fname, fnamejpg))
			#
			phi+=(xmax-x0)/float(phiRes)
		
			iframe+=1
		if B>2:
			B-=2
		else:
			B-=.5
		
	if dosave and False:
		try:
			os.system('mencoder mf:///home/myoder/Documents/Research/Rundle/econoPhysics/econoDensityWave/%s/*.jpg -mf w=800:h=600:fps=10:type=jpg -ovc copy -o %s/%s'% (imagesdir, imagesdir, movieName))
		except:
			print('mencoder not installed or mencoder call not correct. you should still be able to compile your movie from stills.')
			
	return None
#
def johnsstilframes(out_path='stills'):
	# another version of still frames for john.
	# TODO: this is pretty micro-scripted. generalize the script, loop over params, etc.
	fnum=0
	# dwaveFrames(B=10., phi=0., ballpos=None, fignum=0, Yrange=None):
	fnamebase='stillframe'
	if not os.path.isdir(out_path): os.makedirs(out_path)
	#
	dwaveFrames(10, 2., [-7,-3], fnum, [-13, 75])
	plt.savefig('stills/%s00.png' % fnamebase)
	os.system('convert stills/%s00.png stills/%s00.jpg' % (fnamebase, fnamebase))
	#
	dwaveFrames(10, 0., [-7,7], fnum, [-13, 75])
	plt.savefig('stills/%s01.png' % fnamebase)
	os.system('convert stills/%s01.png stills/%s01.jpg' % (fnamebase, fnamebase))
	#
	dwaveFrames(10, -2., 'left', fnum, [-13, 75])
	plt.savefig('stills/%s02.png' % fnamebase)
	os.system('convert stills/%s02.png stills/%s02.jpg' % (fnamebase, fnamebase))
	#
	dwaveFrames(10, 2., 'right', fnum, [-13, 75])
	plt.savefig('stills/%s03.png' % fnamebase)
	os.system('convert stills/%s03.png stills/%s03.jpg' % (fnamebase, fnamebase))
	#
	dwaveFrames(1., 2., None, fnum, [-13, 75])
	plt.savefig('stills/%s04.png' % fnamebase)
	os.system('convert stills/%s04.png stills/%s04.jpg' % (fnamebase, fnamebase))	

def johnsMovies(dosave=True,figsize=(6,5), legend_font_size=10):
	# def doDwaveMovie3(B0=10, dosave=False, imagesdir='images1', lbls=['Stable asset value', 'Metastable asset value'], movieName='dwavemovie1.avi'):
	doDwaveMovie(B0=10, dosave=dosave, imagesdir='imagesEq', lbls=['Locked fault', 'Metastable fault'], movieName='dwaveEq.avi', x_label='Deficit in Slip', y_label='Potential Energy', figsize=figsize, legend_font_size=legend_font_size )
	doDwaveMovie(B0=10, dosave=dosave, imagesdir='imagesEcon', lbls=['Stable asset value', 'Metastable asset value'], movieName='dwaveEcon.avi', x_label='Deficit in Equity (Leverage)', y_label='Asset Values',figsize=figsize, legend_font_size=legend_font_size)

###########################

	


if __name__=='__main__':
	pass
else:
	plt.ion()
