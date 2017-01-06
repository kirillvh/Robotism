# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:42:37 2016

@author: Kirill Van Heerden
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import math
import quadprog


# x.T*Q*x + q.T*x + r = 0
class Quadratic(object):
    def __init__(self, Q=None, q=None, r=0.0):
        assert Q is None or type(Q) is np.ndarray
        assert q is None or type(q) is np.ndarray
        self.n = 0 if q is None else max(q.shape)
        self.Q = Q if Q is not None else np.zeros((self.n, self.n))
        self.q = q if q is not None else np.zeros((self.n))
        self.r = r

    def evalQuad(self, x):
        """ Evaluates x^T Q x + q^T*x + r
        """
        return (x.T.dot(self.Q.dot(x))) + (self.q.dot(x)) + self.r

    def __add__(self, other):
        return Quadratic(Q=self.Q+other.Q, q=self.q+other.q, r=self.r+other.r)

    def __sub__(self, other):
        return Quadratic(Q=self.Q-other.Q, q=self.q-other.q, r=self.r-other.r)

    def __neg__(self):
        return Quadratic(Q=-self.Q, q=-self.q, r=-self.r)

    def __pos__(self):
        return self
    def __str__(self):
        return "Q:"+os.linesep+str(self.Q)+os.linesep+"q: "+str(self.q)+os.linesep+"r: "+str(self.r)
    def __call__(self, x):
        return self.evalQuad(x)

class QP(object):
    """ Quadratic Program
        min : x'*Q*x + q'*x + r
        st  : p'*x + r = 0 ,  i=[0,...,ne] equality constraints
            : h'*x + r < 0 ,  i=[0,...,ni] inequality constraints
    """
    def __init__(self, goal=None, constraintsEq=None, constraintsIq=None):
        assert constraintsEq == None or type(constraintsEq) == list
        assert constraintsIq == None or type(constraintsIq) == list
        assert goal == None or type(goal) == Quadratic
        self.goal = Quadratic() if goal is None else goal
        #equality constraints
        self.constraintsEq = [] if constraintsEq is None else constraintsEq
        #inequality constraints
        self.constraintsIq = [] if constraintsIq is None else constraintsIq
        self.n = 0 if goal is None else goal.n

    def setGoal(self,Q,q):
        self.goal.Q=Q
        self.goal.q=q
        self.n=len(q)

    def evalGoal(self, x):
        assert type(x) == np.matrix
        return goal.evalQuad(x)

    def __str__(self):
        s = "QCQP, goal eq:"+os.linesep
        s += str(self.goal)+os.linesep
        s += str(len(self.constraintsEq))+" equality constraints: "+os.linesep
        for c in self.constraintsEq: s += str(c)+os.linesep
        s += os.linesep
        s += str(len(self.constraintsIq))+" inequality constraints: "
        for c in self.constraintsIq: s += str(c)+os.linesep
        return s

    def CeqMat(self):
        if len(self.constraintsEq)<=0: return []
        ret = None
        for c in self.constraintsEq:
            if ret is None:
                ret = c.q
            else: ret = np.vstack([ret, c.q])
        return ret

    def DeqVec(self):
        if len(self.constraintsEq)<=0: return []
        ret = None
        for c in self.constraintsEq:
            if ret is None:
                ret = c.r
            else: ret = np.hstack([ret, c.r])
        return ret

    def CiqMat(self):
        if len(self.constraintsIq)<=0: return []
        ret = None
        for c in self.constraintsIq:
            if ret is None:
                ret = c.q
            else: ret = np.vstack([ret, c.q])
        return ret

    def DiqVec(self):
        if len(self.constraintsIq)<=0: return []
        ret = None
        for c in self.constraintsIq:
            if ret is None:
                ret = c.r
            else: ret = np.hstack([ret, c.r])
        return ret
    def solve(self, solver="quadprog"):
        if solver=="quadprog":
#==============================================================================
# quadprog format
# min: 0.5*x'*G*X - aT*x
# s.t: C'*X >= b
#==============================================================================
            assert self.goal.Q != []
            assert self.goal.q != []

            Q = 2.0*self.goal.Q
            q = -self.goal.q
            Ceq = self.CeqMat()
            deq = self.DeqVec()
            Ciq = self.CiqMat()
            diq = self.DiqVec()
            C = []
            d = []
            if Ceq != []:
                C = np.vstack([Ceq, -Ceq])
                d = -np.vstack([-deq, deq])
            if Ciq != [] and C != []:
                C = np.vstack([C, -Ciq])
                d = np.vstack([d, diq])
            elif Ciq != [] and C == []:
                C = -Ciq
                d = diq

            if C==[]: # unconstrained path
                x,f,xu,i,lagr,iact= quadprog.solve_qp(Q, q)
                return x
            else: # constrained path
                x,f,xu,i,lagr,iact= quadprog.solve_qp(Q, q, C.T, d.T)
                print("constrained::::::::::")
                return x
def test():
    G = Quadratic(Q=np.array([[1.0, 0.0], [0.0, 1.0]]), q=2*np.array([-2.0, -2.0]), r=4.0)
    C = Quadratic(q=np.array([[1.0], [0.0]]), r=-1.0)
    y = np.array([10.0, 10.0])
    prob = QP(goal=G, constraintsEq = [C])
    print(prob)
    print("Solution")
    print(prob.solve())

if __name__ == "__main__":
    test()