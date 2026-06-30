import random
import numpy

class EvoNode:
    def __init__(self, condition=None, seed=None):
        self.seed = seed            # random seed for reproducibility and vector encoding
        random.seed(seed)           # set the random seed for reproducibility
        numpy.random.seed(seed)           # set the random seed for reproducibility

        self.condition = condition          # lambda function for the branch condition
        self.c_str = ""                  # string representation of the condition
        self.feat = None                    # feature index
        self.feat_range = ()

        self.left = None                    # either a Node or a class label
        self.right = None                   # either a Node or a class label
        self.value = None
        self.is_right = False
        self.is_root = False
        
    def set_seed(self,seed,debug=False):
        # convert seed to int if it is a hex string
        if isinstance(seed, str):
            seed = int(seed, 16)
        self.seed = seed
        random.seed(seed)           # set the random seed for reproducibility
        numpy.random.seed(seed)

        # confirm the seed is set
        if debug:
            print(f"Seed {self.seed}")
            print(f"Random: {random.random()}")
            print(f"Numpy: {numpy.random.random()}")

    def is_leaf(self):
        return self.value is not None
    
    def __str__(self):
        side = "F"
        if self.is_right:
            side = "T"
        elif self.is_root:
            side = ""

        return f"({side}) [{self.c_str if self.value is None else self.value}]"
    

    def clone(self):
        ''' Returns a deep copy of the node '''
        clone = EvoNode(self.condition)
        clone.condition = self.condition
        clone.c_str = self.c_str
        clone.feat = self.feat
        clone.feat_range = self.feat_range
        clone.left = self.left.clone() if self.left else None
        clone.right = self.right.clone() if self.right else None
        clone.value = self.value
        clone.is_right = self.is_right
        clone.is_root = self.is_root
        return clone

    def make_branches(self, split_chance, full_x,full_y, poss_feat, vec=None):
        ''' Creates the left and right branches of the decision tree 
            Either creates a leaf and returns a class label or creates a new Node
        '''

        # evaluate node
        if self.is_leaf():
            return [None, None]  # no branches to create
        
        lx,ly, rx,ry = self.getMatchData(full_x, full_y)  # get the data points that satisfy the branch condition

        if len(ly) == 0 or len(ry) == 0:  # no data points to split
            self.value = numpy.bincount(ly).argmax() if len(ly) > 0 else numpy.bincount(ry).argmax()
            return [None, None]

        res = [-1, -1]  # return values for the left and right branches
        og_seed = self.seed

        # left branch
        if self.left is None:
            self.left = EvoNode()
            lposs = poss_feat.copy()  # copy the available features for the left branch

            if vec is not None:
                self.set_seed(vec.pop(0))
            else:
                self.set_seed(og_seed-1 if og_seed else None)
            #self.set_seed(og_seed-1 if og_seed else None)

            # no data left to split, so turn into a leaf
            if len(lposs) == 0 or len(set(ly)) == 1:  
                self.left.value = random.choice(full_y)   
                #self.left.value = numpy.bincount(ly).argmax() # majority class of the left branch

            # turn into leaf
            elif random.random() > split_chance:          # create a leaf node with majority class
                self.left.value = random.choice(full_y)    # majority class of the left branch
                #self.left.value = numpy.bincount(ly).argmax()

            # otherwise split again
            else:                                       # split again
                condition, cstr, feat, feat_range = self.make_condition(lx,lposs)  # create a condition for the branch
                self.left.condition = condition                           # create a new node
                self.left.c_str = cstr
                self.left.feat = feat
                self.left.feat_range = feat_range

                # remove the feature column as a possibility
                lposs.remove(feat)                  # remove the feature column from x
                res[0] = lposs
                
        if self.right is None:
            self.right = EvoNode()
            self.right.is_right = True
            rposs = poss_feat.copy()  # copy the available features for the left branch

            if vec is not None:
                self.set_seed(vec.pop(0))
            else:   
                self.set_seed(og_seed+1 if og_seed else None)

            # no data left to split, so turn into a leaf
            if len(rposs) == 0 or len(set(ry)) == 1:   
                self.right.value = random.choice(full_y)    # majority class of the right branch
                # self.right.value = numpy.bincount(ry).argmax()    # majority class of the right branch

            # turn into leaf 
            elif random.random() > split_chance:          # create a leaf node with majority class
                self.right.value = random.choice(full_y)
                # self.right.value = numpy.bincount(ry).argmax()    # majority class of the right branch

            # otherwise split again
            else:                                       # split again
                condition, cstr, feat, feat_range = self.make_condition(rx,rposs)  # create a condition for the branch
                self.right.condition = condition                           # create a new node
                self.right.c_str = cstr
                self.right.feat = feat
                self.right.feat_range = feat_range

                # remove the feature column as a possibility
                rposs.remove(feat)                  # remove the feature column from x
                res[1] = rposs

        return res

    def make_condition(self, x, poss_feat):
        ''' Creates a condition for the branch based on the feature and threshold '''
        # randomly pick a feature to split on and a threshold
        if len(poss_feat) == 0:
            raise ValueError("No features left to split on.")
        
        self.set_seed(self.seed)

        feat = random.choice(poss_feat)
        
        feat_range = (numpy.min(x[:,feat]), numpy.max(x[:,feat]))  # min and max of the feature
        threshold = random.uniform(feat_range[0], feat_range[1])  # random threshold between min and max of the feature
        
        # create a split branch lambda function; left if true, right if false
        condition = lambda d: d[feat] < threshold
        c_str = f"x[{feat}] < {threshold}"  # string representation of the condition

        return condition, c_str, feat, feat_range
    

    def getMatchData(self, x, y):
        ''' Returns the data points that satisfy the branch condition '''
        left_indices = []       # false matches
        right_indices = []      # true matches
        for i in range(len(x)):
            if not self.condition(x[i]):        # greater than or equal to (F)
                left_indices.append(i)
            else:                           # less than (T)
                right_indices.append(i)

        return x[left_indices], y[left_indices], x[right_indices], y[right_indices]
    
    def same_node(self, other):
        ''' Helper function to check if two nodes are the same '''
        if self.is_leaf() and other.is_leaf():     # both leaves
            return self.value == other.value

        return self.c_str == other.c_str    # check if the conditions are the same  
    


class EvoDecTree:
    def __init__(self, X, y):
        self.train_X = X
        self.train_y = y
        self.tree = None
        self.split_chance = 0.7


    def __str__(self):
        ''' Returns a string representation of the decision tree '''
        return self.node_str(self.tree, 0)
    
    def clone(self):
        ''' Returns a clone of the decision tree '''
        clone = EvoDecTree(self.train_X, self.train_y)
        clone.tree = self.tree.clone()  # clone the tree
        return clone

    def node_str(self,n,d=0):
        ''' Returns a string representation of the node '''
        if n is None:
            return ""

        if n.is_leaf():
            return ("\t"*d)+ "- "+str(n)
        else:
            return ("\t"*d)+ "- "+str(n) +"\n" + self.node_str(n.left,d+1) + "\n"+self.node_str(n.right,d+1)

    def build_tree(self):
        ''' Builds the decision tree based on the training data and evolves the branches randomly '''

        # create the root split node
        root = EvoNode()
        root.is_root = True
        all_features = list(range(self.train_X.shape[1]))
        c,s,f,r = root.make_condition(self.train_X, all_features)
        all_features.remove(f)  # remove the feature column from x
        root.condition = c
        root.c_str = s
        root.feat = f
        root.feat_range = r
        root.value = None

        self.tree = root
        stack = [{"n":root, "x":self.train_X, "y":self.train_y, "f":all_features}]

        # create the full tree
        #i = 0
        #while len(stack) > 0 and i < 1000:
        while len(stack) > 0:
            node_set = stack.pop(0)
            node = node_set["n"]
            x = node_set["x"]
            y = node_set["y"]
            poss_feat = node_set["f"]

            #print(f"Node: {i}, {node.c_str} | {poss_feat} features left")

            # create the branches (if possible)
            if node and not node.is_leaf():
                p = node.make_branches(self.split_chance, x, y, poss_feat, vec=None)
                l,r = p
                #print(f"Left: {l}, Right: {r}")

                if l == -1 and r == -1:
                    # both branches are leaves, so skip and continue down the stack
                    pass
                else:
                    lx,ly, rx,ry = node.getMatchData(x, y)
                    if l is not None:
                        stack.append({"n":node.left, "x":lx, "y":ly, "f":l})
                    if r is not None:
                        stack.append({"n":node.right, "x":rx, "y":ry, "f":r})

            #i += 1


    def build_tree_from_vec(self, vec):
        ''' Builds the decision tree based on the training data and evolves the branches randomly 
            setting the seed at each node to the vector value

            vec: length of X features ^ 2 (4 bit hexadecimal) (worst case: branch -> branch + branch)
        '''

        # create the root split node
        root = EvoNode()
        seed = vec.pop(0)  # get the first seed value from the vector
        root.set_seed(seed)
        root.is_root = True
        all_features = list(range(self.train_X.shape[1]))
        c,s,f,r = root.make_condition(self.train_X, all_features)
        all_features.remove(f)  # remove the feature column from x
        root.condition = c
        root.c_str = s
        root.feat = f
        root.feat_range = r
        root.value = None

        self.tree = root
        stack = [{"n":root, "x":self.train_X, "y":self.train_y, "f":all_features}]

        # create the full tree
        #i = 0
        #while len(stack) > 0 and i < 1000:
        while len(stack) > 0:
            node_set = stack.pop(0)
            seed_val = vec.pop(0)  # get the next seed value from the vector

            node = node_set["n"]
            x = node_set["x"]
            y = node_set["y"]
            poss_feat = node_set["f"]

            # set the seed for the node
            node.set_seed(seed_val)

            #print(f"Node: {i}, {node.c_str} | {poss_feat} features left")

            # create the branches (if possible)
            if node and not node.is_leaf():
                p = node.make_branches(self.split_chance, x, y, poss_feat, vec=vec)
                l,r = p
                #print(f"Left: {l}, Right: {r}")

                if l == -1 and r == -1:
                    # both branches are leaves, so skip and continue down the stack
                    pass
                else:
                    lx,ly, rx,ry = node.getMatchData(x, y)
                    if l is not None:
                        stack.append({"n":node.left, "x":lx, "y":ly, "f":l})
                    if r is not None:
                        stack.append({"n":node.right, "x":rx, "y":ry, "f":r})

            #i += 1

    def predict(self, X):
        ''' Predicts the class labels for the input data '''
        if self.tree is None:
            raise ValueError("Tree has not been built yet.")

        predictions = []
        for x in X:
            node = self.tree
            while node and not node.is_leaf():
                if node.condition(x):
                    node = node.right
                else:
                    node = node.left

            predictions.append(node.value)

        return numpy.array(predictions)

    def accuracy(self, X, y):
        ''' Returns the accuracy of the model on the test data '''
        if self.tree is None:
            raise ValueError("Tree has not been built yet.")

        y_true = numpy.array(y).reshape(-1,1)
        y_pred = self.predict(X)
        tp = numpy.sum((y_pred == 1) & (y_true == 1))
        fp = numpy.sum((y_pred == 1) & (y_true == 0))
        fn = numpy.sum((y_pred == 0) & (y_true == 1))
        f1 = 2 * tp / (2 * tp + fp + fn + numpy.finfo(float).eps)
        return f1
    

    def proportion(self, X1, X2):
        ''' Returns the proportion of predictions '''
        if self.tree is None:
            raise ValueError("Tree has not been built yet.")

        probx = self.predict(X1).mean()
        proby = self.predict(X2).mean()
        return probx, proby
    
    def grab_X_split(self, X, c):
        ''' Grabs the splits of X based on the comparisons '''
        if "GENDER" in c:
            male = X[X['gender']==1]
            female = X[X['gender']==0]
            return male.to_numpy(), female.to_numpy()

        elif "AGE" in c:
            old = X[X['age_binary']==1]
            young = X[X['age_binary']==0]
            return old.to_numpy(), young.to_numpy()

        elif "RACE" in c:
            white = X[X['race']==1]
            other = X[X['race']==0]
            return white.to_numpy(), other.to_numpy()

    def proportion2(self, X, comp):
        ''' Returns the proportion of predictions '''
        if self.tree is None:
            raise ValueError("Tree has not been built yet.")
        numerator, denominator = self.grab_X_split(X, comp)
        numerator = numpy.array(numerator)
        denominator = numpy.array(denominator)
        prob = (self.predict(numerator).mean()) / (self.predict(denominator).mean() + numpy.finfo(float).eps)


        # if prob > 2:
        #     print(comp)
        #     print(f"Proportion: {prob}")
        #     print(f"Numerator size: {len(numerator)} [avg: {self.predict(numerator).mean()}]")
        #     print(f"Denominator size: {len(denominator)} [avg: {self.predict(denominator).mean()}]")

        return prob

    def run(self, X, Xnp, y, comps):
        return self.accuracy(Xnp,y), (self.proportion2(X, comps[0]), self.proportion2(X, comps[1]), self.proportion2(X, comps[2]))


    def mutate(self, mutation_rate=0.2):
        ''' Mutates the decision tree by randomly changing the conditions '''
        if self.tree is None:
            raise ValueError("Tree has not been built yet.")

        stack = [self.tree]
        while len(stack) > 0:
            node = stack.pop(0)
            if node and not node.is_leaf():
                # mutate the condition with a certain probability
                if random.random() < mutation_rate:
                    
                    # randomly condition cut off based on the feature range
                    feat = node.feat
                    feat_range = node.feat_range
                    threshold = random.uniform(feat_range[0], feat_range[1])
                    node.condition = lambda d: d[feat] < threshold
                    node.c_str = f"x[{feat}] < {threshold}"


                stack.append(node.left)
                stack.append(node.right)


    def same_tree(self, other):
        ''' Checks if two trees are the same '''
        if self.tree is None or other.tree is None:
            return False

        stackA = [self.tree]
        stackB = [other.tree]

        while len(stackA) > 0 and len(stackB) > 0:
            nodeA = stackA.pop(0)
            nodeB = stackB.pop(0)


            if nodeA is None and nodeB is None:  # both nodes are None
                continue
 
            # check conditions if both branches
            if not nodeA.same_node(nodeB):
                print(f"Node A: {nodeA.c_str if nodeA.is_leaf() == False else nodeA.value}")
                print(f"Node B: {nodeB.c_str if nodeB.is_leaf() == False else nodeB.value}")
                print(f"Seeds: {nodeA.seed}(a) {nodeB.seed}(b)")
                return False

            stackA.append(nodeA.left)
            stackA.append(nodeA.right)
            stackB.append(nodeB.left)
            stackB.append(nodeB.right)

        return len(stackA) == 0 and len(stackB) == 0        # all nodes have been checked
    
        
