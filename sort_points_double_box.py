#!/usr/bin/python

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil

import numpy as np
import math
from optparse import OptionParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from scipy import interpolate 
import io

# parse command line arguments

parser = OptionParser()

parser.add_option("--u3", dest="u3file",
                  default="2d-u3.txt", help="read 2D u3 traces from FILE", metavar="FILE")

parser.add_option("--u23", dest="u23file",
                  default="2d-u2-minus-u3.txt", help="read 2D u2-u3 traces from FILE", metavar="FILE")

(options, args) = parser.parse_args()

# output_file writes the redistibuted points into separate files for
# each isolevel

def output_file(matrix,i,fnbase):
    with open('sorted_points_'+fnbase+'_iso_%s.txt' % i,'w') as outputfile:
        np.savetxt(outputfile, matrix)

# sorted_points sorts the points in one iso.

def sorted_points(points):
    # algorithm: sort according to y value
    col=1
    return points[np.argsort(points[:,col])]

# sort_and_output takes the data read from a COMSOL output file,
# splits into isos, sorts, outputs to file and returns the same
# structure with the sorted points in it.

def sort_and_output(d,fnbase):

    unique=np.unique(d[2]) # get the unique values of the isolevels

    for i in range(len(unique)):
        value=unique[i]
        print("Processing iso level %f"%value)
        iso=d[:,(d[2]==value)]  # all the points in this iso
        points=np.transpose(iso[:2])  # np array with all the points in this iso
        this_sorted_points=sorted_points(points)

        x,y=np.transpose(this_sorted_points)
        plt.plot(x,y)

        iso_level=np.full((len(this_sorted_points),1),value)
        thisd=np.concatenate((this_sorted_points,iso_level),axis=1)

        if i==0:
            alld=thisd
        else:
            alld=np.concatenate((alld,thisd))

        # save individual isos
        output_file(this_sorted_points,value,fnbase)

    alld=np.transpose(alld)

    with open('sorted_'+fnbase+'.dat','w') as outputfile:
        np.savetxt(outputfile,alld)

    return alld


fig=plt.figure

with open(options.u3file) as stream:
    du3=np.loadtxt(stream,comments="%",unpack=True)

alldu3=sort_and_output(du3,'du3')

with open(options.u23file) as stream:
    du23=np.loadtxt(stream,comments="%",unpack=True)

alldu23=sort_and_output(du23,'du23')

#allu3u23=np.concatenate((alldu3,alldu23))

#with open('sortedPoints.dat') as inputfile:
#    d2=np.loadtxt(inputfile)
#p=np.transpose(allu3u23)
#X,Y,Z=alldu23

#plt.plot(X,Y)

plt.title('Extracted Traces')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.savefig('extracted_traces.png')
plt.show()
