'''
 Current Folder: INBOX   	Sign Out
Compose   Addresses   Folders   Options   Search   Help   	SquirrelMail

Search Results | Delete		Forward | Forward as Attachment | Reply | Reply All
Subject:   	Movie in ipython
From:   	"James Holliday" <jrholliday@ucdavis.edu>
Date:   	Fri, August 27, 2010 1:58 pm
To:   	yoder@physics.ucdavis.edu
Priority:   	Normal
Options:   	View Full Header |  View Printable Version  | Download this as a file
'''

import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#

class Usage(Exception):
    def __init__(self, msg):
        self.msg = 'Error: %s\n' % msg

#-----------------------------------------------------------------------------#

class Plotter(object):
    plt.ion()

    def __init__(self, start, stop, beta, bigev=[]):
        self.start = start
        self.stop  = stop

        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_title('Beta = %s' % beta)
        self.line1 = self.ax.plot_date( [], [], '-r', label='M<=3.3')[0]
        self.line2 = self.ax.plot_date( [], [], '-b', label='M<=3.6')[0]
        self.ax.legend(loc=0, numpoints=1)
        self.fig.autofmt_xdate()

        for event in bigev:
            if (self.start <= event <= self.stop):
                self.ax.axvline(x=pylab.date2num(event), color='k')

    def __del__(self):
        plt.show()

    def updateData(self, X, Y1, Y2):
        self.line1.set_data(pylab.date2num(X), Y1)
        self.line2.set_data(pylab.date2num(X), Y2)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_xlim((pylab.date2num(self.start),
                          pylab.date2num(self.stop)))
        self.fig.canvas.draw()

#-----------------------------------------------------------------------------#

def readCatalog(filename, bounds=(-360,-90,360,90), big=6.0):
    catalog = []
    bigev = []

    # Open the catalog file
    file = open(filename)

    # Is it a pickle file?
    try:
        catalog = pickle.load(file)
        bigev = pickle.load(file)
    except:
        # Nibble the header data
        line = file.readline()
        line = file.readline()

        # Parse the catalog file
        for line in file.readlines():
            try:
                # Read in full Date-Time specification
                timestamp = dateutil.parser.parse(line[0:22])
            except:
                # Something broke.  Probabily bad Time specification
                timestamp = dateutil.parser.parse(line[0:10])

            lat = float(line[24:31])
            lng = float(line[32:41])
            mag = float(line[50:54])

            # Check that we're within our bounds
            if (bounds[0]<=lng<=bounds[2]) and (bounds[1]<=lat<=bounds[3]):
                # Save the event
                catalog.append((timestamp,mag))

                # Save all large events
                if mag >= big:
                    bigev.append(timestamp)

    # Close the catalog file
    file.close()

    # End readCatalog(...)
    return sorted(catalog, reverse=True), sorted(bigev)

#-----------------------------------------------------------------------------#

def saveCatalog(filename, catalog, bigev):
    file = open(filename, 'w')
    pickle.dump(catalog, file)
    pickle.dump(bigev, file)
    file.close()

    # End saveCatalog(...)
    return None

#-----------------------------------------------------------------------------#

def countback(catalog, mag, beta=1.0, big=6.0,
date=datetime.datetime.today()):
    count = 0
    end = dateutil.parser.parse('1932/1/1')

    # Parse the date, if needed
    if date.__class__ is str:
        date = dateutil.parser.parse(date)

    # Use the GR b-value to calculate the countback number
    num = 10**(-beta*(mag-big))

    # Loop over all the events
    for event in catalog:

        # Are we in the past?  Is the event large enough?
        if (event[0] < date) and (event[1] >= mag):
            count += 1

            # Have we reached enough events?
            if (count >= num):
                end = event[0]
                break

    # End countback(...)
    return (date-end).days

#-----------------------------------------------------------------------------#

def snapshot(cat, beta=1.0, big=6.0, date=datetime.datetime.today()):
    windows = [
        countback(cat, 0.1*mag, beta, big, date) for mag in xrange(30,37)
        ]

    # End snapshot(...)
    return numpy.array( windows )

#-----------------------------------------------------------------------------#

def makeTimeseries(cat,
                   bigev=[],
                   beta=1.0,
                   big=6.0,
                   start=dateutil.parser.parse('1960/1/1'),
                   stop=datetime.datetime.today(),
                   dt=30,
                   plot=True
                   ):

    # Parse the dates, if needed
    if start.__class__ is str:
        start = dateutil.parser.parse(start)
    if stop.__class__ is str:
        stop = dateutil.parser.parse(stop)

    X  = []
    Y1 = []
    Y2 = []

    if plot is True:
        myFig = Plotter(start,stop,beta,bigev)

    date = start
    while date < stop:
        ts = snapshot(cat, beta, big, date)

        X.append(date.date())
        Y1.append( ts[0:10*(3.4-3.0)].mean() )
        Y2.append( ts[0:10*(3.7-3.0)].mean() )
        date += datetime.timedelta(days=dt)

        if plot is True:
            myFig.updateData(X,Y1,Y2)

    # End makeTimeseries(...)
    return tuple(zip(X, Y1, Y2))

#-----------------------------------------------------------------------------#

def main(argv=None):
    import optparse
    import re

    # Do it this way to trap any changes to sys.argv
    if argv is None: argv = sys.argv

    # Enter the "main loop"
    try:
        # Handle option parsing
        usage = 'usage: %prog [options] CATALOG'
        description = 'SOVIX - Seismic Omori Volatility IndeX'
        version = '%prog 0.9'
        parser = optparse.OptionParser(usage=usage,
                                       description=description,
                                       version=version)
        parser.add_option('-s', '--start',
                          action='store', type='string', dest='start',
                          default='1/1/1980', metavar='Y/M/D',
                          help='set start date [default 1/1/1980]')
        parser.add_option('-S', '--stop',
                          action='store', type='string', dest='stop',
                          default='1/1/2009', metavar='Y/M/D',
                          help='set stop date [default 1/1/2009]')
        parser.add_option('-t', '--dt',
                          action='store', type='int', dest='dt',
                          default='30', metavar='VAL',
                          help='set timestep (in days) [default 30]')
        parser.add_option('-b', '--beta',
                          action='store', type='float', dest='beta',
                          default=1.0, metavar='VAL',
                          help='set value of beta [default 1.0]')
        parser.add_option('-r', '--region',
                          action='store', type='string', dest='bounds',
                          default='-360/360/-90/90', metavar='A/B/C/D',
                          help='set region boundaries: xmin,xmax,ymin,ymax')
        parser.add_option('-m', '--magnitude',
                          action='store', type='float', dest='big',
                          default=6.0, metavar='VAL',
                          help='set target magnitude [default 6.0]')
        parser.add_option('-p', '--pickle',
                          action='store', type='string', dest='savepickle',
                          default='', metavar='FILE',
                          help='pickle the catalog data after parsing')
        parser.add_option('-q', '--quiet',
                          action='store_false', dest='showplot',
                          default=True,
                          help='do not generate/display graph')

        (options, args) = parser.parse_args()

        # Parse catalog filename
        catfile = args[:1]

        # Check Options
        if catfile == []:
            raise Usage('No value for CATALOG')
        else:
            catfile = catfile[0]

        options.bounds = [float(i) for i in re.split(r'[,/ ]',
options.bounds)]
        if len(options.bounds) is not 4:
            raise Usage('REGION requires 4 values: xmin,xmax,ymin,ymax')

        ### BEGIN ACTUAL CODE !
###############################################
        print 'Loading catalog...'
        cat, bigev = readCatalog(catfile, options.bounds, options.big)

        if options.savepickle != '':
            print 'Pickling catalog data...'
            saveCatalog(options.savepickle, cat, bigev)

        print 'Making plot...'
        data = makeTimeseries(cat,
                              bigev,
                              options.beta,
                              options.big,
                              options.start,
                              options.stop,
                              options.dt,
                              options.showplot)

        ### End main(...)
#####################################################
        return 0

    except Usage, err:
        print >>sys.stderr, err.msg
        parser.print_help()
        return 1

#-----------------------------------------------------------------------------#

if __name__ == '__main__':
    sys.exit(main())



