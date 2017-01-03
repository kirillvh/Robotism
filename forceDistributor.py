# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 01:22:40 2017

@author: Kirill Van Heerden
"""

import numpy as np

class forceDistributor(object):
    def __init__(self):
        #                 COM coordinates
        #
        # F_z1------F_z2        .       F_z5------F_z6
        # |            |     x /|\      |            |
        # | Left foot  |        |____\  | Right foot |
        # |            |           y /  |            |
        # F_z3------F_z4                F_z7------F_z8
        #
        # Now lets define the relations
        # sum(F_z) = mg , The combined F_z's must equal to the robots gravity force
        # (A~)*F_z = [tau_x,tau_y] , Let tau be the torque generated by F_z
        # A~ = [x1...x8;y1...y8] where xi,yi are the positions of the F_zi indexes
        # Then combine it:
        # [A~; 1] = [tau_x,tau_y;mg]
        # A*F_z = [cop_x*mg;cop_y*mg;mg] , remember tau_x = cop_x*mg, cop is center of pressure
        self.A=np.array([[1, 1, -1, -1, 1, 1, -1, -1], [-2, -1, 1, 2, -2, -1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1]])

    def solve(self,mg=9.8, cop_x=0, cop_y=0):
        b = np.array([[cop_x*mg],[cop_y*mg],[mg]])
        # may need to solve under defined system, thus least squares solver may be needed
        # Remember, using matrix inverse to solve linear systems is usually
        # computationaly expensive and has poor accuracy, use linear solver instead if possible
        f_z=np.linalg.lstsq(self.A, b)[0]
        return f_z

    def reconstructAMatrix(self,x,y,weightRight=1.,weightLeft=1.):
        lastrow = [weightRight,weightRight,weightRight,weightRight,\
                   weightLeft,weightLeft,weightLeft,weightLeft]
        self.A = np.array([x,y,lastrow])

    def FzLeft(self,Fz):
        return Fz[0]+Fz[1]+Fz[2]+Fz[3]

    def FzRight(self,Fz):
        return Fz[4]+Fz[5]+Fz[6]+Fz[7]

if __name__ == "__main__":
    fd = forceDistributor()

    print("COM between two feet")
    print(fd.solve(mg=10., cop_x=0., cop_y=0))

    fd.reconstructAMatrix([1,1,-1,-1,1,1,-1,-1],[-1,1,-1,1,3,4,3,4],weightRight=1.,weightLeft=1.)
    print("COM above right foot with left foot active")
    print(fd.solve(mg=10., cop_x=0., cop_y=0))

    fd.reconstructAMatrix([1,1,-1,-1,1,1,-1,-1],[-1,1,-1,1,3,4,3,4],weightRight=1.,weightLeft=0.)
    print("COM above right foot with left foot inactive")
    print(fd.solve(mg=10., cop_x=0., cop_y=0))

    Fz=fd.solve(mg=10., cop_x=0., cop_y=0)
    Fzl=fd.FzLeft(Fz)