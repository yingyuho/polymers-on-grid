class QList:
	def __init__(self):
		self.list = []
		self.n = 0
	def append(self, obj):
		self.list.append(obj)
		obj.__qid = self.n
		self.n += 1
	def remove(self, obj):
		self.n -= 1
		self.list[obj.__qid] = self.list.pop()
		self.list[obj.__qid].__qid = obj.__qid