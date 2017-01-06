# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:43:39 2017

@author: Kirill Van Heerden
"""

from QP import *
import numpy as np
import matplotlib.pyplot as plt

class QPTrajectory(object):
    def __init__(self, initalState=[0.,0.,0.],samplingTime=0.1, COMHeight=0.5,numtrjPoints=16):
        self.initalState = np.array(initalState)
        self.samplingTime = samplingTime
        self.COMHight=0.5
        self.numtrjPoints = numtrjPoints
        self.A = np.array([[1.0, samplingTime, 0.5*samplingTime*samplingTime],\
                           [0.,1.,samplingTime], [0.,0.,1.]])
        self.B = np.array([[(1./6.)*samplingTime*samplingTime*samplingTime],\
        [0.5*samplingTime*samplingTime],[samplingTime]])
        # ZMP/COP selection vector
        self.Cz = np.array([1.,0.,-COMHeight/9.81])
        # Acceleration selection vector
        self.Ca = np.array([0.,0.,1.])
        # Velocity selection vector
        self.Cv = np.array([0.,1.,0.])
        # Position selection vector
        self.Cp = np.array([1.,0.,0.])

        self.Pzs = self.constructPSmat(self.Cz)
        self.Pzu = self.constructPUmat(self.Cz)

        self.Pas = self.constructPSmat(self.Ca)
        self.Pau = self.constructPUmat(self.Ca)

        self.Pvs = self.constructPSmat(self.Cv)
        self.Pvu = self.constructPUmat(self.Cv)

        self.Pps = self.constructPSmat(self.Cp)
        self.Ppu = self.constructPUmat(self.Cp)

        self.qp=QP()

    def constructPSmat(self,C, dbg=False):
        ps = np.zeros((self.numtrjPoints,3))
        for i in range(self.numtrjPoints):
            temp=C.dot(np.linalg.matrix_power(self.A,i+1))
            for j in range(3):
                ps[i][j] = temp[j]
        return ps

    def constructPUmat(self,C,dbg=False):
        pu = np.zeros((self.numtrjPoints, self.numtrjPoints))
        for i in range(self.numtrjPoints): # 0,1,...,numtrjPoints-1
            for j in range(self.numtrjPoints): # 0,1,...,numtrjPoints-1
                #this can be optimized...
                pu[i][j]=C.dot(np.linalg.matrix_power(self.A,i-j).dot(self.B))
                if j>=i: break
        return pu

    def generateGoalFunction(self,comTrj,alpha=0.5,beta=10.0,gamma=1.0, R=1E-3):
        Q = 0.5*R*np.identity(self.numtrjPoints)+\
        0.5*alpha*self.Pvu.T.dot(self.Pvu)+\
        0.5*beta*self.Ppu.T.dot(self.Ppu)

        q = alpha*self.Pvu.T.dot(self.Pvs.dot(self.initalState))+\
        beta*self.Ppu.T.dot(self.Pps.dot(self.initalState))+\
        -beta*self.Ppu.T.dot(comTrj)

        self.qp.setGoal(Q,q)

    def generateCOPconstraints(self,cop_max,cop_min):
        constraintUp = Quadratic(q=self.Pzu,r=-(-self.Pzs.dot(self.initalState)+cop_max))
        self.qp.constraintsIq.append(constraintUp)

        constraintDown = Quadratic(q=self.Pzu,r=-(-self.Pzs.dot(self.initalState)+cop_min))
        self.qp.constraintsIq.append(-constraintDown)

    def solveJerks(self):
        jerk = self.qp.solve()
        return jerk

    def jerkToZMP(self,jerk):
        return self.Pzs.dot(self.initalState)+self.Pzu.dot(jerk)
    def jerkToAcc(self,jerk):
        return self.Pas.dot(self.initalState)+self.Pau.dot(jerk)
    def jerkToVel(self,jerk):
        return self.Pvs.dot(self.initalState)+self.Pvu.dot(jerk)
    def jerkToPos(self,jerk):
        return self.Pps.dot(self.initalState)+self.Ppu.dot(jerk)

def test():
    N = 16*5
    dt=0.1
    trj_y=QPTrajectory(samplingTime=dt,numtrjPoints=N)
    index = [i*dt for i in range(N)]

    ref = [float(int(round(math.sin(y*0.2)))) for y in range(N)]
    com_yTrjRef = np.array(ref)
    cop_max = 1.2
    cop_min =-1.2
    cop_yMax = np.array([cop_max for i in range(N)])
    cop_yMin = np.array([cop_min for i in range(N)])

    trj_y.generateCOPconstraints(cop_yMax,cop_yMin)

    trj_y.generateGoalFunction(com_yTrjRef)

    jerk_y=trj_y.solveJerks()
    acc_y=trj_y.jerkToAcc(jerk_y)
    vel_y=trj_y.jerkToVel(jerk_y)
    pos_y=trj_y.jerkToPos(jerk_y)
    zmp_y=trj_y.jerkToZMP(jerk_y)


    ax = plt.subplot(2,2,1)
    plt.plot(index,zmp_y,label="zmp")
    plt.plot(index,cop_yMax,label="ZMP-Max")
    plt.plot(index,cop_yMin,label="ZMP-Min")
    plt.plot(index,pos_y,label="pos_y")
    plt.plot(index,ref,label="ref_y")
    ax.set_xlabel("s")
    ax.set_ylabel("m")
    legend = ax.legend(loc='lower right', shadow=True, prop={'size':10})

    ax = plt.subplot(2,2,2)
    ax.plot(index,vel_y,label="vel")
    ax.plot(index,acc_y,label="acc")
    ax.set_xlabel("s")
    ax.set_ylabel("m/s and m/s^2")
    legend = ax.legend(loc='upper right', shadow=True, prop={'size':10})

    #now can do the same for another axis
    #trj_x=QPTrajectory()
    #but lets rather see what happens with stricter ZMP constraints

    trj_y2=QPTrajectory(samplingTime=dt,numtrjPoints=N)

    cop_max = 1.01 # if this is too strict, then solver can fail
    cop_min =-1.01
    cop_yMax2 = np.array([cop_max for i in range(N)])
    cop_yMin2 = np.array([cop_min for i in range(N)])

    trj_y2.generateCOPconstraints(cop_yMax2,cop_yMin2)

    trj_y2.generateGoalFunction(com_yTrjRef)

    jerk_y2=trj_y2.solveJerks()
    acc_y2=trj_y2.jerkToAcc(jerk_y2)
    vel_y2=trj_y2.jerkToVel(jerk_y2)
    pos_y2=trj_y2.jerkToPos(jerk_y2)
    zmp_y2=trj_y2.jerkToZMP(jerk_y2)

    ax = plt.subplot(2,2,3)
    plt.plot(index,zmp_y2,label="zmp")
    plt.plot(index,cop_yMax2,label="ZMP-Max")
    plt.plot(index,cop_yMin2,label="ZMP-Min")
    plt.plot(index,pos_y2,label="pos_y")
    plt.plot(index,ref,label="ref_y")
    ax.set_xlabel("s")
    ax.set_ylabel("m")
    legend = ax.legend(loc='lower right', shadow=True, prop={'size':10})

    ax = plt.subplot(2,2,4)
    ax.plot(index,vel_y2,label="vel")
    ax.plot(index,acc_y2,label="acc")
    ax.set_xlabel("s")
    ax.set_ylabel("m/s and m/s^2")
    legend = ax.legend(loc='upper right', shadow=True, prop={'size':10})

if __name__ == "__main__":
    test()
