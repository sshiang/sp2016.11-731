import sys
import os
import copy
from nltk.tokenize import wordpunct_tokenize
from zss import simple_distance

class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or list()

    @staticmethod
    def get_children(node):
        """
        Default value of ``get_children`` argument of :py:func:`zss.distance`.
        :returns: ``self.children``.
        """
        return node.children

    @staticmethod
    def get_label(node):
        """
        Default value of ``get_label`` argument of :py:func:`zss.distance`.
        :returns: ``self.label``.
        """
        return node.label

    def addkid(self, node, before=False):
        """
        Add the given node as a child of this node.
        """
        if before:  self.children.insert(0, node)
        else:   self.children.append(node)
        return self

    def get(self, label):
        """:returns: Child with the given label."""
        if self.label == label: return self
        for c in self.children:
            if label in c: return c.get(label)

    def iter(self):
        """Iterate over this node and its children in a preorder traversal."""
        queue = collections.deque()
        queue.append(self)
        while len(queue) > 0:
            n = queue.popleft()
            for c in n.children: queue.append(c)
            yield n

    def __contains__(self, b):
        if isinstance(b, str) and self.label == b: return 1
        elif not isinstance(b, str) and self.label == b.label: return 1
        elif (isinstance(b, str) and self.label != b) or self.label != b.label:
            return sum(b in c for c in self.children)
        raise TypeError, "Object %s is not of type str or Node" % repr(b)

    def __eq__(self, b):
        if b is None: return False
        if not isinstance(b, Node):
            raise TypeError, "Must compare against type Node"
        return self.label == b.label

    def __ne__(self, b):
        return not self.__eq__(b)

    def __repr__(self):
        return super(Node, self).__repr__()[:-1] + " %s>" % self.label

	def __str__(self):
		s = "%d:%s" % (len(self.children), self.label)
		s = '\n'.join([s]+[str(c) for c in self.children])
		return s


def readNode(line, parentNode):

	#print line
	index_left = [i for i in range(len(line)) if line[i]=="("]#line.index("(")
	index_right = [i for i in range(len(line)) if line[i]==")"]

	#print "i",index_left, index_right

	if len(index_left)==0 and len(index_right)==0:
		seg = line.split(" ")
		word = line.strip()
		chrunk = ""
		if len(seg)==2:
			word = seg[1]
			chrunk = seg[0]
			# print chrunk
		return Node(word,[])

	indexs = index_left+index_right
	indexs.sort()
	level = 0
	remember = []
	for idx in indexs:
		# get first level bracket only.
		if idx in index_left:
			level +=1
			if level == 1:
				start = idx
		else:
			level -=1
			if level == 0:
				remember.append((start,idx))

		'''
		if idx in index_left and level == 0:
			start = idx
			level +=1
		elif idx in index_left and level != 0:
			level += 1
		elif idx in index_right and level == 0:
			remember.append((start,idx))
			level -=1
		elif idx in index_right and level != 0:
			level -=1
		'''		

	#print "r", remember
	name = line[:index_left[0]]
	node = Node(name,[])
	#print "n",name
	for rem in remember:
		#print "o", line[rem[0]+1:rem[1]]
		c_node = readNode(line[rem[0]+1:rem[1]],node)
		if c_node != None:
			node.children.append(c_node)
	#print "!", node.children
	return node


'''
def traverse(nodes):
	toVisit = [copy.deepcopy(nodes)]
	terminals = []
	while len(toVisit)!=0:
		target = toVisit[-1]
	
		#print "t", target.children
		for child in target.children:
			#print "c",child
			if child.words!="":
				terminals.append(child.words)
			else:
				#print child
				toVisit.append(child)
		del toVisit[0]
	print "\t",terminals
	return terminals
'''

def traverse(nodes):
	toVisit = [copy.deepcopy(nodes)]
	terminals = []
	while len(toVisit)!=0:
		target = toVisit[0]
		if len(target.children)==0:
			terminals.append(target)
			del toVisit[0]
			continue
		#print "t", target.children
		for child in target.children:
			#print "c",child
			if len(child.children)==0:
				terminals.append(child)
			else:
				#print child
				toVisit.append(child)
		del toVisit[0]
	#print "\t",terminals
	return terminals


def dumpFile(splitNum,toSplit,f):
	#f.write("%d "%splitNum)
	for i in range(len(toSplit)):
		children = traverse(toSplit[i])
		f.write(" ".join([str(i) for x in range(len(children))])+" ")

#length = [len(line.strip().split(" ")) for line in open("data/german")]

#f = open("data/tree_german_processed","w")
count = 0

start = int(sys.argv[1])
end = int(sys.argv[2])

#textFile = sys.argv[1]

textFile = "data/train-test.hyp1-hyp2-ref"
treeFile = "data/text_tree"
plainFile = "data/text"

outFile = "output_%d_%d.txt"%(start,end)
out2File = "score_%d_%d.txt"%(start,end)

treeDict = {}

trees = [line.strip() for line in open(treeFile)]
plains = [line.strip() for line in open(plainFile)]

assert (len(trees) == len(plains)), "length mismatch"

for i in range(len(trees)):
	treeDict[plains[i]] = trees[i]

f = open(outFile,"w")
fx = open(out2File, "w")


lines = [line.strip() for line in open(textFile)]


#for line in open(textFile):
for i in range(start, min(end, len(lines))):

	if i%100 == 0:
		print i


	line = lines[i]
	candidates = line.strip().split("|||")

	ref = " ".join([x.encode('utf-8') for x in wordpunct_tokenize(candidates[2].strip().decode("utf-8"))])
	hyp1 = " ".join([x.encode('utf-8') for x in wordpunct_tokenize(candidates[0].strip().decode("utf-8"))])
	hyp2 = " ".join([x.encode('utf-8') for x in wordpunct_tokenize(candidates[1].strip().decode("utf-8"))])
	
	ref = treeDict[ref] #candidates[2].strip()[2:-2]
	hyp1 = treeDict[hyp1] #candidates[0].strip()[2:-2]
	hyp2 = treeDict[hyp2] #candidates[1].strip()[2:-2]


	#print ref
		
	rootRef = readNode(ref, None)
	rootHyp1 = readNode(hyp1, None)
	rootHyp2 = readNode(hyp2, None)

	#print rootRef.label

	score1 = simple_distance(rootRef, rootHyp1)
	score2 = simple_distance(rootRef, rootHyp2)

	#print score1
	fx.write("%d\t%d\n"%(score1,score2))

	if score1 >= score2:
		f.write("1\n")
	else:
		f.write("-1\n")

f.close()
fx.close()
