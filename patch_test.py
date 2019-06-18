# Fri May 24 10:11:45 CDT 2019 Jeff added this line.
# June 16, 2019 Jeff modified to limit to class definitions.

from scipy.constants import mu_0, pi
import numpy as np
from patch import *

# Halliday & Resnick, 10th ed., question 29.13
p0 = np.array([0,0,0])
p1 = np.array([0.18,0,0])
r = np.array([0.09,0.131,0])
i = 0.0582
print(b_segment(i,p0,p1,r))

# Halliday & Resnick, 10th ed., question 29.13
p0 = np.array([0,0,0])
p1 = np.array([0.18,0,0])
x=0.09
y=0.131
z=0
i = 0.0582
print(b_segment_2(i,p0,p1,x,y,z))

# Halliday & Resnick, 10th ed., question 29.17
p0 = np.array([0,0,0])
p1 = np.array([0.136,0,0])
r = np.array([0.136,0.251,0])
i = 0.693
print(b_segment(i,p0,p1,r))

# Halliday & Resnick, 10th ed., question 29.17
p0 = np.array([0,0,0])
p1 = np.array([0.136,0,0])
x=0.136
y=0.251
z=0
i = 0.693
print(b_segment_2(i,p0,p1,x,y,z))

# Halliday & Resnick, 10th ed., question 29.31
a = 0.047
i = 13.0
p0 = np.array([0,0,0])
p1 = np.array([2*a,0,0])
p2 = np.array([2*a,a,0])
p3 = np.array([a,a,0])
p4 = np.array([a,2*a,0])
p5 = np.array([0,2*a,0])
points = (p0,p1,p2,p3,p4,p5)
r = np.array([2*a,2*a,0])
print(b_loop(i,points,r))

# Halliday & Resnick, 10th ed., question 29.83
a = 0.08
i = 10.0
p0 = np.array([0,0,0])
p1 = np.array([a,0,0])
p2 = np.array([a,-a,0])
p3 = np.array([0,-a,0])
points = (p0,p1,p2,p3)
r = np.array([a/4,-a/4,0])
print(b_loop(i,points,r))

# repeat of 29.83:
thiscoil = coil(points,i)
print(thiscoil.b(r))
thiscoil.set_current(i/2.0)
print(thiscoil.b(r))
print(thiscoil.b_prime(a/4,-a/4,0))


# stupid test
p0 = np.array([-0.15935262,.7,-.9])
p1 = np.array([-0.15935262,.7,+.9])
x=0.0
y=0.7
z=0.8
i = 0.1
print(b_segment(i,p0,p1,np.array([x,y,z])))
print(b_segment_2(i,p0,p1,x,y,z))
x=0.2
y=0.7
z=0.8
i = 0.1
print(b_segment(i,p0,p1,np.array([x,y,z])))
print(b_segment_2(i,p0,p1,x,y,z))
