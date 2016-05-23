# -*- coding=utf-8 -*-
import numpy as np
import random
class SVM(object):
	def __init__(self,train_data,test_data=None,C=200,toler=0.0001,kernel="rbf",sigma=1,eta=1000):
		"""
		:param train_data:
		:param test_data:
		:param C:
		:param toler:
		:param kernel: kernel function ,valid data :"rbf","liner"
		:return:
		"""
		self.x=train_data[:,:-1]
		self.y=train_data[:,-1]
		if test_data is not None:
			self.test_x=test_data[:,:-1]
			self.test_y=test_data[:,-1]
		else:
			self.test_x=None
			self.test_y=None
		self.C=C
		self.toler=toler
		self.kernel=kernel
		self.sigma=sigma
		self.m,self.n=self.x.shape
		self.K=self.cal_kernel(self.x,self.sigma)
		self.alpha=np.zeros((self.m,1))
		self.b=0
		self.w=np.zeros((self.m,1))
		self.eta=eta
		self.errorCache=np.zeros((self.m,2))#first column is valid flag
		self.support_vectors=None
		self.support_vectors_lables=None
		self.support_vectors_num=0
		#calculate the kernel matrix
	def Kernel(self,x,xi,sigma):
		m,n=x.shape
		K=np.zeros((m,1))
		if self.kernel=="linear":
			K=x.dot(xi.transpose())
		elif self.kernel=="rbf":

			for j in range(m):
				temp=x[j]-xi
				K[j]=temp.dot(temp.transpose())
			K=np.exp(K/(-2*sigma**2))
		else :
			print "wrong kernel ,please input linear or rnf"
		return K
	def cal_kernel(self,x,sigma):
		"claculate the kernel matrix"
		m,n=x.shape
		K=np.zeros((m,m))
		for i in range(m):
			K[:,i]=self.Kernel(x,x[i],sigma).transpose()
		return  K

	def innerLoop(self, alpha_i):
		"the inner loop for optimizing alpha i and alpha j"
		error_i = self.cal_error(alpha_i)
		"""
	    check and pick up the alpha who violates the KKT condition
		satisfy KKT condition
		 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
		 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
		 3) yi*f(i) <= 1 and alpha == C (between the boundary)
		 violate KKT condition
		 because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
		 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
		 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
		 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
		"""

		if (self.y[alpha_i] * error_i < -self.toler and self.alpha[alpha_i,0] < self.C) or \
				(self.y[alpha_i] * error_i > self.toler and self.alpha[alpha_i,0] > 0):

			# step 1: select alpha j
			alpha_j, error_j = self.select_alpha_j(alpha_i, error_i)
			alpha_i_old = self.alpha[alpha_i,0]
			alpha_j_old = self.alpha[alpha_j,0]

			# step 2: calculate the boundary L and H for alpha j
			if self.y[alpha_i] != self.y[alpha_j]:
				L = max(0, self.alpha[alpha_j,0] - self.alpha[alpha_i,0])
				H = min(self.C, self.C + self.alpha[alpha_j,0] - self.alpha[alpha_i,0])
			else:
				L = max(0, self.alpha[alpha_j,0] + self.alpha[alpha_i,0] - self.C)
				H = min(self.C, self.alpha[alpha_j,0] + self.alpha[alpha_i,0])
			if L == H:
				return 0

			# step 3: calculate eta (the similarity of sample i and j)
			eta = 2.0 * self.K[alpha_i, alpha_j] - self.K[alpha_i, alpha_i] \
					  - self.K[alpha_j, alpha_j]
			if eta >= 0:
				return 0

			# step 4: update alpha j
			self.alpha[alpha_j,0] -= self.y[alpha_j] * (error_i - error_j) / eta

			# step 5: clip alpha j
			if self.alpha[alpha_j,0] > H:
				self.alpha[alpha_j,0] = H
			if self.alpha[alpha_j,0] < L:
				self.alpha[alpha_j,0] = L

			# step 6: if alpha j not moving enough, just return
			if abs(alpha_j_old - self.alpha[alpha_j,0]) < 0.00001:
				#self.update_error( alpha_j)
				self.alpha[alpha_j,0]=alpha_j_old
				return 0

			# step 7: update alpha i after optimizing aipha j
			self.alpha[alpha_i,0] += self.y[alpha_i] * self.y[alpha_j] \
									* (alpha_j_old - self.alpha[alpha_j,0])

			# step 8: update threshold b
			b1 = self.b - error_i - self.y[alpha_i] * (self.alpha[alpha_i,0] - alpha_i_old)* self.K[alpha_i,alpha_i] \
				- self.y[alpha_j] * (self.alpha[alpha_j,0] - alpha_j_old)* self.K[alpha_i, alpha_j]
			b2 = self.b - error_j - self.y[alpha_i] * (self.alpha[alpha_i,0] - alpha_i_old) * self.K[alpha_i, alpha_j] \
				- self.y[alpha_j] * (self.alpha[alpha_j,0] - alpha_j_old) * self.K[alpha_j, alpha_j]
			if (0 < self.alpha[alpha_i,0]) and (self.alpha[alpha_i,0] < self.C):
				self.b = b1
			elif (0 < self.alpha[alpha_j,0]) and (self.alpha[alpha_j,0] < self.C):
				self.b = b2
			else:
				self.b = (b1 + b2) / 2.0

			# step 9: update error cache for alpha i, j after optimize alpha i, j and b
			self.update_error( alpha_j)
			self.update_error(alpha_i)

			return 1
		else:
			return 0
	def cal_error(self,i):
		"calcuate the  output error when we input x_i to svm"
		g_xi=(self.alpha[:,0]*self.y).dot(self.K[:,i])+self.b
		return g_xi-self.y[i]

	def update_error(self, alpha_k):
		"update the error cache for alpha k after optimize alpha k"
		error = self.cal_error( alpha_k)
		self.errorCache[alpha_k] = [1, error]


	def select_alpha_j(self, alpha_i, error_i):
		"select alpha j which has the biggest step"
		self.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)
		candidateAlphaList = np.nonzero(self.errorCache[:, 0])[0]
		maxStep = 0; alpha_j = 0; error_j = 0

		# find the alpha with max iterative step
		if len(candidateAlphaList) > 1:
			for alpha_k in candidateAlphaList:
				if alpha_k == alpha_i:
					continue
				error_k = self.cal_error(alpha_k)
				if abs(error_k - error_i) > maxStep:
					maxStep = abs(error_k - error_i)
					alpha_j = alpha_k
					error_j = error_k
		# if came in this loop first time, we select alpha j randomly
		else:
			alpha_j = alpha_i
			while alpha_j == alpha_i:
				alpha_j = int(random.uniform(0, self.m))
			error_j = self.cal_error(alpha_j)

		return alpha_j, error_j

	def SMO(self):
		"platt's Sequential Minimal Optimization algorithm"
		iter=0;alphaPairsChanged=0;entireSet=True
		while (iter < self.eta) and ((alphaPairsChanged > 0) or (entireSet)):
			alphaPairsChanged = 0
			if entireSet:   #go over all
				for i in range(self.m):
					alphaPairsChanged += self.innerLoop(i)
					print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
				iter += 1
			else:#go over non-bound (railed) alphas
				nonBoundIs = np.nonzero((self.alpha > 0) * (self.alpha< self.C))[0]
				for i in nonBoundIs:
					alphaPairsChanged += self.innerLoop(i)
					print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
				iter += 1
			if entireSet: entireSet = False #toggle entire set loop
			elif (alphaPairsChanged == 0): entireSet = True
			print "iteration number: %d" % iter
	def cal_w(self):
		self.w=(self.alpha[:,0]*self.y).dot(self.x)
		print self.w.shape
		print self.w
	def fit(self):
		self.SMO()
		self.cal_w()

		"find the support vectors"
		index=np.nonzero(self.alpha>0)[0]
		self.support_vectors=self.x[index , :]
		self.support_vectors_lables=self.y[index]
		self.support_vectors_alpha=self.alpha[index,0]
		print "there are %d Support Vectors" % len(self.support_vectors)
	def predict(self,xi):
		"predict the input x_i .positive class is 1 and negtive class is -1"
		kernel=self.Kernel(self.support_vectors,xi,self.sigma)
		return np.sign((self.support_vectors_alpha*self.support_vectors_lables).transpose().dot(kernel)+self.b)
	def accuracy(self):
		num=0.0
		for i in range(self.m):
			if self.predict(self.x[i])==np.sign(self.y[i]):
				num+=1
		print "the training data accuracy is %f"%(num/self.m)
		if self.test_x is not None:
			num=0.0
			for i in range(len(self.test_x)):
				if self.predict(self.test_x[i])==np.sign(self.test_y[i]):
					num+=1
			print "the test data accuracy is %f"%(num/len(self.test_x))
def loadData(path):
	data=np.loadtxt(path)
	return data
training_data=loadData("D:\\SelfLearning\\Machine Learning\\MachineLearningInAction\\machinelearninginaction\\Ch06\\testSetRBF.txt")
test_data=loadData("D:\\SelfLearning\\Machine Learning\\MachineLearningInAction\\machinelearninginaction\\Ch06\\testSetRBF2.txt")
svm=SVM(training_data,test_data=test_data,sigma=1.3,C=200,kernel="rbf")
svm.fit()
svm.accuracy()

