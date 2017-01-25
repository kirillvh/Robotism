import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv

class kalmanKinematic2:
	def __init__(self, encoder_sampl_period, encResolution,uncertaintyGain):
		self.T = encoder_sampl_period
		self.encResolution = encResolution
		self.x = np.zeros((2,1))
		self.P = np.zeros((2,2))
		self.A = np.zeros((2,2))
		self.A[0][0] = 1
		self.A[0][1] = self.T
		self.A[1][0] = 0
		self.A[1][1] = 1
		self.B = np.zeros((2,1))
		self.C = np.zeros((1,2))
		self.C[0][0] = 1.0
		self.C[0][1] = 0.0
		self.B[0] = 0.5*self.T*self.T
		self.B[1] = self.T
		self.R = self.encResolution*self.encResolution/12.0
		self.Q = np.zeros((2,2))
		self.Q[0][0] = (1.0/3.0)*self.T*self.T*self.T
		self.Q[0][1] = (1.0/2.0)*self.T*self.T
		self.Q[1][0] = (1.0/2.0)*self.T*self.T
		self.Q[1][1] = self.T
		self.Q = self.Q*uncertaintyGain
	def update(self,thetaMeasured, ddthetaModel):
		#A-priori update
		self.x = self.A.dot(self.x) + self.B*ddthetaModel;
		self.P = self.A.dot(self.P).dot(self.A.T) + self.Q;
		#A-posteriori update
		y = np.identity(1)*thetaMeasured - self.C.dot(self.x);
		S = self.C.dot(self.P).dot(self.C.T) + self.R;
		K = self.P.dot(np.transpose(self.C)).dot(inv(S));
		self.x = self.x + K.dot(y);
		self.P = (np.identity(2) - K.dot(self.C)).dot(self.P);
		return self.x

class kalmanKinematic3:
#this class only needs the encoder data
	def __init__(self, encoder_sampl_period, encResolution,uncertaintyGain):
		self.T = encoder_sampl_period
		self.encResolution = encResolution
		self.x = np.zeros((3,1))
		self.P = np.zeros((3,3))
		self.A = np.zeros((3,3))
		self.A[0][0] = 1.0
		self.A[0][1] = self.T
		self.A[0][2] = 0.5*self.T*self.T
		self.A[1][0] = 0
		self.A[1][1] = 1.0
		self.A[1][2] = self.T
		self.A[2][0] = 0
		self.A[2][1] = 0
		self.A[2][2] = 1.0
		self.B = np.zeros((3,1))
		self.C = np.zeros((1,3))
		self.C[0][0] = 1.0 #position sense
		self.C[0][1] = 0.0
		self.C[0][2] = 0.0
		self.B[0] = 0.5*self.T*self.T
		self.B[1] = self.T
		self.B[2] = 1.0
		self.R = self.encResolution*self.encResolution/12.0
		self.Q = np.zeros((3,3))
		self.Q[0][0] = (1.0/20.0)*self.T*self.T*self.T*self.T*self.T
		self.Q[0][1] = (1.0/8.0)*self.T*self.T*self.T*self.T
		self.Q[0][2] = (1.0/6.0)*self.T*self.T*self.T
		self.Q[1][0] = (1.0/8.0)*self.T*self.T*self.T*self.T
		self.Q[1][1] = (1.0/3.0)*self.T*self.T*self.T
		self.Q[1][2] = (1.0/2.0)*self.T*self.T
		self.Q[2][0] = (1.0/6.0)*self.T*self.T*self.T
		self.Q[2][1] = (1.0/2.0)*self.T*self.T
		self.Q[2][2] = self.T

		#self.Q[0][0] = (1.0/3.0)*self.T*self.T*self.T
		#self.Q[0][1] = (1.0/2.0)*self.T*self.T
		#self.Q[1][0] = (1.0/2.0)*self.T*self.T
		#self.Q[1][1] = self.T
		self.Q = self.Q*uncertaintyGain
	def update(self,thetaMeasured):
		#A-priori update
		self.x = self.A.dot(self.x) + self.B*ddthetaModel;
		self.P = self.A.dot(self.P).dot(self.A.T) + self.Q;
		#A-posteriori update
		y = np.identity(1)*thetaMeasured - self.C.dot(self.x);
		S = self.C.dot(self.P).dot(self.C.T) + self.R;
		K = self.P.dot(self.C.T).dot(inv(S));
		#print(K)
		self.x = self.x + K*y;
		self.P = (np.identity(3) - K.dot(self.C)).dot(self.P);
		return self.x


class kalmanKinematic3B:
#this class can use both encoder data and dynamic model acceleration
#if no dynamic model is availible, then you MUST substitute the previous acceleration result instead of the dynamic model
	def __init__(self, encoder_sampl_period, encResolution,uncertaintyGain):
		self.T = encoder_sampl_period
		self.encResolution = encResolution
		self.x = np.zeros((3,1))
		self.P = np.zeros((3,3))
		self.A = np.zeros((3,3))
		self.A[0][0] = 1.0
		self.A[0][1] = self.T
		self.A[0][2] = 0.5*self.T*self.T
		self.A[1][0] = 0
		self.A[1][1] = 1.0
		self.A[1][2] = self.T
		self.A[2][0] = 0
		self.A[2][1] = 0
		self.A[2][2] = 1.0

		self.A2 = np.zeros((3,3))
		self.A2[0][0] = 1.0
		self.A2[0][1] = self.T
		self.A2[0][2] = 0#0.5*self.T*self.T
		self.A2[1][0] = 0
		self.A2[1][1] = 1.0
		self.A2[1][2] = 0#self.T
		self.A2[2][0] = 0
		self.A2[2][1] = 0
		self.A2[2][2] = 0#1.0

		self.B = np.zeros((3,1))
		self.C = np.zeros((1,3))
		self.C[0][0] = 1.0 #position sense
		self.C[0][1] = 0.0
		self.C[0][2] = 0.0
		self.B[0] = 0.5*self.T*self.T
		self.B[1] = self.T
		self.B[2] = 1.0
		self.R = self.encResolution*self.encResolution/12.0
		self.Q = np.zeros((3,3))
		self.Q[0][0] = (1.0/20.0)*self.T*self.T*self.T*self.T*self.T
		self.Q[0][1] = (1.0/8.0)*self.T*self.T*self.T*self.T
		self.Q[0][2] = (1.0/6.0)*self.T*self.T*self.T
		self.Q[1][0] = (1.0/8.0)*self.T*self.T*self.T*self.T
		self.Q[1][1] = (1.0/3.0)*self.T*self.T*self.T
		self.Q[1][2] = (1.0/2.0)*self.T*self.T
		self.Q[2][0] = (1.0/6.0)*self.T*self.T*self.T
		self.Q[2][1] = (1.0/2.0)*self.T*self.T
		self.Q[2][2] = self.T

		#self.Q[0][0] = (1.0/3.0)*self.T*self.T*self.T
		#self.Q[0][1] = (1.0/2.0)*self.T*self.T
		#self.Q[1][0] = (1.0/2.0)*self.T*self.T
		#self.Q[1][1] = self.T
		self.Q = self.Q*uncertaintyGain
	def update(self,thetaMeasured, ddthetaModel):
		#A-priori update
		self.x = self.A2.dot(self.x) + self.B*ddthetaModel;
		self.P = self.A.dot(self.P).dot(self.A.T) + self.Q;
		#A-posteriori update
		y = np.identity(1)*thetaMeasured - self.C.dot(self.x);
		S = self.C.dot(self.P).dot(self.C.T) + self.R;
		K = self.P.dot(self.C.T).dot(inv(S));
		#print(K)
		self.x = self.x + K*y;
		self.P = (np.identity(3) - K.dot(self.C)).dot(self.P);
		return self.x

def main():
	#generate data
	print("Generating data")
	w1 = 10
	w2 = 1
	dt = 0.001 #100us
	encoder_sampl_period = 0.001
	encResolution = 2.0*3.141592/(1024)
	t_period = 5# 10 seconds
	#print(jerks)
	pos = np.zeros((t_period/dt,1))
	vel = np.zeros((t_period/dt,1))
	velM = np.zeros((t_period/dt,1))
	accM = np.zeros((t_period/dt,1))
	acc = np.zeros((t_period/dt,1))
	jerk = np.zeros((t_period/dt,1))
	for i in range(1,int(t_period/dt)):
		pos[i] = math.sin(w1*i*dt) - 2.0*math.cos(w2*i*dt)
		vel[i] = w1*math.cos(w1*i*dt) + 2.0*w2*math.sin(w2*i*dt)
		acc[i] = -w1*w1*math.sin(w1*i*dt) + 2.0*w2*w2*math.cos(w2*i*dt)
		jerk[i] = -w1*w1*w1*math.cos(w1*i*dt) - 2.0*w2*w2*w2*math.sin(w2*i*dt)


	#calculate discretized position signals and vel,acc by finite difference method
	enc = np.zeros((t_period/encoder_sampl_period,1))
	for i in range(1,int(t_period/encoder_sampl_period)):
		enc[i] = pos[i*encoder_sampl_period/dt]
		#discretize
		enc[i] = int(enc[i]/encResolution)*encResolution
		velM[i] = (enc[i]-enc[i-1])/encoder_sampl_period
		accM[i] = (velM[i]-velM[i-1])/encoder_sampl_period



	#use kalman filter
	x = np.zeros((2,1))

	vel_kalman = np.zeros((t_period/dt,1))
	acc_kalman = np.zeros((t_period/dt,1))
	w_filtered = 0
	#P = 0#encoder_sampl_period
	P = np.zeros((2,2))

	kal1 = kalmanKinematic3B(encoder_sampl_period,encResolution,1.0);

	for i in range(1,int(t_period/encoder_sampl_period)):
		noise = np.random.uniform(low=-1.0, high=1.0, size=1)
		#use acceleration model
		x = kal1.update(enc[i],acc[i]+noise)
		#no model acceleration, feedback kalman state
		#x = kal1.update(enc[i],acc_kalman[i-1])
		#print(x)
		vel_kalman[i] = x[1]
		acc_kalman[i] = x[2]

     #plot comparision
	ax=plt.subplot(131)
	plt.plot(enc, label="encoder pos.")
	plt.plot(pos,label="kalman pos.")
	plt.ylabel('rad')
	plt.xlabel('time')
	legend = ax.legend(loc='lower right', shadow=True, prop={'size':10})

	ax=plt.subplot(132)
	plt.plot(vel, label="real vel")
	plt.plot(velM,label="finite diff. vel")
	plt.plot(vel_kalman, label="kalman vel.")
	plt.ylabel('rad/sec')
	plt.xlabel('time')
	legend2 = ax.legend(loc='lower right', shadow=True, prop={'size':10})
	ax.set_ylim([-15,15])

	ax=plt.subplot(133)
	plt.plot(accM,label="finite diff. acc")
	plt.plot(acc, label="real acc.")
	plt.plot(acc_kalman,label="kalman acc.")
	plt.ylabel('rad/sec^2')
	plt.xlabel('time')
	legend2 = ax.legend(loc='lower right', shadow=True, prop={'size':10})
	ax.set_ylim([-10000,10000])
	plt.show()
###
def kalman_velModel(x, P, thetaMeasured, ddthetaModel, T, thetaM, q):
	#do kalman stuff
	A = np.zeros((2,2))
	A[0][0] = 1
	A[0][1] = T
	A[1][0] = 0
	A[1][1] = 1
	B = np.zeros((2,1))
	C = np.zeros((2,1))
	C[0] = 1.0
	B[0] = 0.5*T*T
	B[1] = T
	#R = np.zeros((1,1))
	R = thetaM*thetaM/12.0
	Q = np.zeros((2,2))
	Q[0][0] = (1.0/3.0)*T*T*T
	Q[0][1] = (1.0/2.0)*T*T
	Q[1][0] = (1.0/2.0)*T*T
	Q[1][1] = T

	S = np.zeros((1,1))
	#A-priori update
	x = A*x + B*ddthetaModel;
	P = A*P*A.transpose() + Q;

	#A-posteriori update
	y = np.identity(1)*thetaMeasured - C*x;
	S = C*P*np.transpose(C) + R;
	K = P*np.transpose(C)*inv(S);

	x = x + K*y;
	P = (np.identity(2) - K*C)*P;
	print(P)
	return x

if __name__ == "__main__":
    main()
