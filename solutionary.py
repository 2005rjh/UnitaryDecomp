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


class new_solutionary():
	def __init__(self):
		self.sols = {}

	def add(self, sol, key):
		sol.check_consistency()
		self.sols[key] = sol

	def remove_at(self, key):
		assert self.length() > 0
		self.sols.pop(key)

	def save(self, name):
		filehandler = open(name,"wb")
		pickle.dump(self.sols,filehandler)
		filehandler.close()

	def load(self, filename):
		file = open(filename, 'rb')
		self.sols = pickle.load(file)
		file.close()
		for x in self.sols:
			x.check_consistency()

	def access(self, key):
		assert self.length() > 0
		return self.sols[key]

	def length(self):
		return len(self.sols)

	def empty(self):
		self.sols = {}
		
	def reorder_keys(self):
		sols = self.sols.values()
		self.empty()
		for sol in sols:
			print(sol.coef)
			print(sol.str(verbose=3))
			print(sol.Utarget)
			key = input("Enter name: ")
			self.add(sol, key)

	def index(self, index):
		keys = self.sols.keys()
		for key in keys:
			num = ""
			for x in key:
				if x != "-": num+=x
				else: break
			if int(num) == index: return self.sols[key]
		return -1

	def keys(self):
		keys = self.sols.keys()
		return keys

	def print_sols(self):
		output = ""
		keys = self.sols.keys()
		for x in keys: output+=x+" "
		print(output)