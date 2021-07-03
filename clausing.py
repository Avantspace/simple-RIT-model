import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import ode
from scipy import integrate
import csv
import itertools
import datetime, time
import re
import pandas as pd
import math
import random

thickScreen = 0.3
thickAccel = 1.5
rScreen = 0.95
rAccel = 0.6
gridSpace = 0.6
npart = int(1e4)


#Monte Carlo Routine that calculates Clausing factor for CEX
#returns Clausing Factor and Downstream Correction factor


rBottom = rScreen / rAccel
lenBottom = (thickScreen + gridSpace) / rAccel
lenTop = thickAccel / rAccel
Length = lenTop + lenBottom
iescape = 0
maxcount = 0
icount = 0
nlost = 0
vztot = int(0)
vz0tot = int(0)


for ipart in range(1,npart):
    notgone = True
    r0 = rBottom * np.sqrt(random.random())
    z0 = int(0)
    costheta = np.sqrt(1 - random.random())

    if costheta > 0.99999: costheta = 0.99999
    phi = 2 * np.pi * random.random()
    sintheta = np.sqrt(int(1)  - costheta**2)
    vx = np.cos(phi) * sintheta
    vy = np.sin(phi) * sintheta
    vz = costheta
    rf = rBottom
    t = (vx * r0 + np.sqrt((vx**2 + vy**2) * (rf**2) - (vy * r0)**2)) / (vx**2 + vy**2)
    z = z0 + vz * t
    vz0tot = vz0tot + vz
    icount = 0
    while notgone:
        icount = icount + 1
        if (z < lenBottom): # hit wall of bottom cylinder and is re-emitted
            r0 = rBottom
            z0 = z
            costheta = np.sqrt(1 - random.random())
            if(costheta > 0.99999): costheta = 0.99999
            phi = 2 * np.pi * random.random()
            sintheta = np.sqrt(int(1) - costheta ** 2)
            vx = np.cos(phi) * sintheta
            vy = np.sin(phi) * sintheta
            vz = costheta
            rf = rBottom
            t = (vx * r0 + np.sqrt((vx ** 2 + vy ** 2) * (rf ** 2) - (vy * r0) ** 2)) / (vx ** 2 + vy ** 2)
            z = z0 + vz * t

        if((z >= lenBottom) and (z0 < lenBottom)): # ' bottom cylinder re-emission emitted below but going up
            t = (lenBottom - z0) / vz
            r = np.sqrt((r0 - vx * t)**2 + (vy * t)**2)
            if (r <= 1): # continuing upwar
                rf = int(1)  #
                t = (vx * r0 + np.sqrt((vx **2 + vy **2) * (rf **2) - (vy * r0)**2)) / (vx**2 + vy**2)
                z = z0 + vz * t
            else: #hit the upstream side of the accel grid and is re-emitted downward
                r0 = r
                z0 = lenBottom
                costheta = np.sqrt(1 - random.random())
                if (costheta > 0.99999): costheta = 0.99999
                phi = 2 * np.pi * random.random()
                sintheta = np.sqrt(int(1) - costheta ** 2)
                vx = np.cos(phi) * sintheta
                vy = np.sin(phi) * sintheta
                vz = -costheta
                rf = rBottom
                t = (vx * r0 + np.sqrt((vx ** 2 + vy ** 2) * (rf ** 2) - (vy * r0) ** 2)) / (vx ** 2 + vy ** 2)
                z = z0 + vz * t
        if((z >= lenBottom) and (z <= Length)): #hit the upper cylinder wall and is re-emitted
            r0 = int(1)
            z0 = z
            costheta = np.sqrt(1 - random.random())
            if (costheta > 0.99999): costheta = 0.99999
            phi = 2 * np.pi * random.random()
            sintheta = np.sqrt(int(1) - costheta ** 2)
            vx = np.cos(phi) * sintheta
            vy = np.sin(phi) * sintheta
            vz = costheta
            rf = int(1)
            t = (vx * r0 + np.sqrt((vx ** 2 + vy ** 2) * rf ** 2 - (vy * r0) ** 2)) / (vx ** 2 + vy ** 2)
            z = z0 + vz * t

        if(z < lenBottom): # find z when particle hits the bottom cylinder
            rf = rBottom
            if((vx **2 + vy** 2) * (rf ** 2) - (vy * r0)**2 < int(0)):
                t = (vx * r0) / (vx ** 2 + vy ** 2) # if sqr arguement is less than 0 then set sqr term to 0
            else:
                t = (vx * r0 + np.sqrt((vx ** 2 + vy ** 2) * rf ** 2 - (vy * r0) ** 2)) / (vx ** 2 + vy ** 2)
            z = z0 + vz * t

        if(z < 0):
            notgone = False

        if(z > Length):
            iescape = iescape + 1
            vztot = vztot + vz
            notgone = False

        if(icount > 1000):
            notgone = False
            icount = 0
            nlost = nlost + 1

    if(maxcount < icount): maxcount = icount
    #print(f'Particle number:{ipart}, Particle lost:{iescape}')
print(vztot/vz0tot)
R1 = (rBottom ** 2) * iescape / npart
R2 = maxcount
R3 = nlost
vz0av = vz0tot / npart
vzav = vztot / iescape
DenCor = vz0av / vzav # Downstream correction factor
print(R1, R2, R3, vz0av, vzav, DenCor, iescape/npart)