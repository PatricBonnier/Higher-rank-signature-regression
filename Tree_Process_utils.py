import numpy as np
import Sig_utils as Sig

class TreeProcess:  
    def __init__(self, data, leaf = False): 
        """
            Parameters
            ----------
            data: (float, [float], [data]) where data may be of the same type or float 
                data[0] is the value at the root, data[1] is a list of transition probabilites from the root to each of the children in data[2]
            leaf: bool
                if True then the node will be treated as a leaf
        """
        self.children = None
        self.transitions = None
        
        if type(data) is not tuple or leaf:
            self.value = data
            return

        V = data[0]       
        T = data[1]
        C = data[2]
        
        self.value = V
        self.children = []
        self.transitions = []
        
        for i,c in enumerate(C):
            child = TreeProcess(c)
            self.children.append(child)
            self.transitions.append(T[i])
            
    def __str__(self):
        s = ""
        for p in self._generatePaths():
            s += "p = {0}: \n {1} \n\n".format(p[0], str(p[1]))
        return s
            
    def _generatePaths(self): 
        """
            Returns
            -------
            paths: list of tuples of the form (float, numpy array)
                a list of all the possible paths and their corresponding probabilities 
        """
        if self.children == None:
            return [(1,[self.value])]
        paths = []
        for i,c in enumerate(self.children):
            for x in c._generatePaths():
                x[1].insert(0, self.value)
                paths.append((self.transitions[i] * x[0], x[1]))
        return paths

    def sample(self):
        paths = self._generatePaths()
        ind = np.random.choice(range(len(paths)), p = [x[0] for x in paths])
        return paths[ind][1]
    
    def ExpSig(self, N, Augment = False):
        """
            Parameters
            ----------
            N: int
                Truncation parameter, the signature up to level N is returned
            Augment: bool
                If true, a time coordinate is added before computing the signature
        
            Returns
            -------
            R: Tseries
                A tensor series representing the expected signature up to level N
        """             
        if (self.children == None):
            return Sig.Tseries.One() 
        R = Sig.Tseries.Zero()
        
        for i, X in enumerate(self.children):
            increment = np.array([self.value, X.value])
            E = Sig.Sig(increment, N, Augment)
            R += E.__mul__(X.ExpSig(N,Augment), trunc = N) * self.transitions[i]
        return R    
    
    def ExpSigr2(self, N, Augment = False, sigcache = Sig.Tseries.One()):
        """
            Parameters
            ----------
            N: int
                Truncation parameter, the signature up to level N is returned
            Augment: bool
                If true, a time coordinate is added before computing the signature
            sigcache: Tseries
                used for recursive calls to cache the value at the current node

            Returns
            -------
            R: Tseriesr2 or Tseriesr2Aug
                A rank 2 tensor series representing the expected signature of rank 2 up to level N
        """
        tensorClass = Sig.Tseriesr2 if not Augment else Sig.Tseriesr2Aug
        if (self.children == None):
            return Sig.Tseries.One(),tensorClass.One()
        v2 = tensorClass.Zero()
        v1 = Sig.Tseries.Zero()
        valcache = []

        for i, X in enumerate(self.children):
            increment = np.array([self.value, X.value])
            E = Sig.Sig(increment, N, Augment)

            c1, c2 = X.ExpSigr2(N,Augment,sigcache.__mul__(E,trunc=N))
            valcache.append((c1,c2))

            v1 += E.__mul__(c1, trunc = N) * self.transitions[i]

        for i, c in enumerate(self.children):
            c1, c2 = valcache[i]
            siginc = Sig.Sig(np.array([self.value, X.value]), N, Augment)
            nextsig = sigcache.__mul__(siginc, trunc = N)
            increment = np.array([sigcache.__mul__(v1, trunc= N), nextsig.__mul__(c1, trunc=N)])
            v2 += Sig.Sigr2(increment, N, Augment).__mul__(c2, trunc=N) * self.transitions[i]
        return v1, v2