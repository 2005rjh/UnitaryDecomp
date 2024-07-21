from UnitaryChain import *
import pickle
import numpy as np

class solutionary():
	def __init__(self):
		self.sols = []

	def add(self, sol):
		self.sols.append(sol)

	def remove_at(self, index):
		assert self.length() > 0
		self.sols.pop(index)

	def save(self, name):
		filehandler = open(name,"wb")
		pickle.dump(self.sols,filehandler)
		filehandler.close()

	def load(self, filename):
		file = open(filename, 'rb')
		self.sols = pickle.load(file)
		file.close()

	def access(self, index):
		assert self.length() > 0
		return self.sols[index]

	def length(self):
		return len(self.sols)
	