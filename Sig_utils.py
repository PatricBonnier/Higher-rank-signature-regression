import numpy as np
from copy import deepcopy
from math import factorial

def Sig(data, N, Augment = False): 
    """
        Parameters
        ----------
        data: numpy array 
            The shape must be (number of time steps, number of spacial dimensions)
        N: int
            Truncation parameter, the signature up to level N is returned
        Augment: bool
            If true, a time coordinate is added before computing the signature
        
        Returns
        -------
        S: Tseries
            A tensor series representing the signature up to level N
    """

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    if Augment:
        data = np.concatenate((np.expand_dims(np.array(range(len(data))),axis=1), data), axis=1)

    deltas = data[1:,:]-data[:-1,:]
    S = Tseries.One()
    for v in deltas:
        S = S.__mul__(Tseries.fromVec(v).Exp(N), trunc = N)
    return S

def Sigr2(data, N, Augment = False): 
    """
        Parameters
        ----------
        data: numpy array of Tseries
            The shape must be (number of time steps)
        N: int
            Truncation parameter, the signature up to level N is returned
        Augment: bool
            If true, a time coordinate is added before computing the rank 2 signature
        
        Returns
        -------
        S: Tseriesr2 or Tseriesr2Aug
            A tensor series representing the rank 2 signature up to level N, if Augment is true then this is a Tseriesr2Aug
    """

    deltas = []
    for i in range(len(data)-1):
        deltas.append(data[i+1]-data[i])

    tensorClass = Tseriesr2 if not Augment else Tseriesr2Aug
    S = tensorClass.One()
    for t in deltas:
        S = S.__mul__(tensorClass.fromTens(t).Exp(N), trunc = N)
    return S

### Rank 1 Tensor series ###
#region

class Tseries:
    def __init__(self, values, dim):
        """
            Parameters
            ----------
            values: dictionary {int:numpy array}
                the array with key n must have shape (d,)**n
            dim: int
                the dimension of the underlying space
        """

        self.values = values
        self.dim = dim
    degree = property(lambda self : max(self.values))
    
    def __deepcopy__(self, memo):
        d=deepcopy(self.values)
        return type(self)(d, self.dim)
    
    def __str__(self):
        s=""
        for k in self.values.keys():
            s +="{0} : {1} \n".format(k,self.values[k])
        return "Tensor series of dimension = {} and degree = {}. Values: \n".format(self.dim, self.degree) + s

    def __repr__(self):
        return str(self)
    
    def __add__(self, x, sub=False):
        d = {}
        dim = self.dim if self.dim != 0 else x.dim
        for k in set.union(set(self.values.keys()), set(x.values.keys())):
            n = self.values[k] if k in self.values else 0
            m = x.values[k] if k in x.values else 0
            d[k] = n+m if not sub else n-m
        return type(self)(d, dim)

    def __sub__(self, x):
        return self.__add__(x,sub=True)
            
    def __mul__(self, x, trunc = -1): 
        d = {}
        if isinstance(x, float) or isinstance(x, int):
            x = type(self).fromFloat(x)

        dim = self.dim if self.dim != 0 else x.dim
            
        def tensor_multiply(x,y, out=None):
            n = len(x.shape)
            m = len(y.shape)
            it = np.nditer([x, y, out], ['external_loop'],
                    [['readonly'], ['readonly'], ['writeonly', 'allocate']],
                    op_axes=[list(range(n))+[-1]*m,
                             [-1]*n+list(range(m)),
                             None])
            for (a, b, c) in it:
                np.multiply(a, b, out=c)
            return it.operands[2]

        for i in self.values.keys():
            for j in x.values.keys():
                if trunc > 0 and self._degree(i) + self._degree(j) > trunc:
                    continue
                if not self._isDimZero(i) and not self._isDimZero(j):
                    summand=tensor_multiply(self.values[i],x.values[j])
                else:
                    if self._isDimZero(i):
                        summand=x.values[j] * self.values[i][0]
                    else:
                        summand=self.values[i] * x.values[j][0]
                if self._conc(i,j) not in d.keys():
                    d[self._conc(i,j)]=0
                d[self._conc(i,j)]+=summand 
        return type(self)(d, dim)

    def __rmul__(self, x, trunc = -1):
        if isinstance(x, float) or isinstance(x, int):
            return self.__mul__(x,trunc)
        return None

    def __pow__(self, n):
        R = deepcopy(self)
        for i in range(n-1):
            R = R*self
        return R
    
    def Exp(self, N):
        """
            Parameters
            ----------
            N: int 
                truncation parameter
            Returns
            -------
            R: Tseries
                the exponential up to level N
        """
        R = type(self).One()
        for i in range(1,N+1):
            R += (self**i) * (1/factorial(i))
        return R

    def flatten(self):
        """
            Returns
            -------
            ret: numpy array
                A 1D numpy array representation of the tensor series
        """
        ret = np.ones(0)
        deg = self.degree

        for k in self._generateKeys(deg):
            if k in self.values:
                app = self.values[k].flatten()
            else:
                app = np.zeros(self.dim**self._degree(n))
            ret = np.append(ret, app)
        return ret

    ### Virtual methods

    def _generateKeys(self, N):
        return range(N+1)

    def _degree(self, k):
        return k

    def _conc(self, k,l):
        return k+l

    def _isZero(self, k):
        return k==0

    def _isDimZero(self, k):
        return self._isZero(k)

    ### Initialisers
    
    def One():
        return Tseries({0:np.ones(1)}, dim=0)

    def Zero():
        return Tseries({0:np.zeros(1)}, dim=0)
    
    def fromVec(v):
        return Tseries({1:v}, dim=len(v))

    def fromFloat(f):
        return Tseries({0:np.array([f])}, dim = 0)
 
#endregion

### Rank 2 Tensor series ###
#region 

class Tseriesr2(Tseries):
    def __init__(self, values , dim):
        """
            Parameters
            ----------
            values: dictionary {(int,):numpy array}
                the array with key (n_1, ..., n_k) must have shape (d,)**(n_1 + ... + n_k)
            dim: int
                the dimension of the underlying space
        """
        self.values = values
        self.dim = dim
    degree = property(lambda self : max(self._degree(v) for v in self.values.keys()))    

    ### Operations on multi-indices

    def _generateKeys(self, N):
        def _genrecur(N):
            if N == 0:
                return [(0,)]
            ret = []
            for l in range(1,N+1):
                for k in _genrecur(N-l):
                    ret.append(self._conc((l,), k))
            return ret
        return [i for n in range(N+1) for i in _genrecur(n)] 

    def _degree(self, k):
        return sum(k)

    def _conc(self, k,l):
        if self._isZero(k):
            return l
        if self._isZero(l):
            return k
        return k+l

    def _isZero(self, k):
        return k==(0,) #or k==()

    ### Initialisers

    def One():
        return Tseriesr2.fromTens(Tseries.One())

    def Zero():
        return Tseriesr2.fromTens(Tseries.Zero())
    
    def fromVec(v):
        return Tseriesr2.fromTens(Tseries.fromVec(v))

    def fromFloat(f):
        return Tseriesr2.fromTens(Tseries.fromFloat(f))

    def fromTens(T):
        d = {}
        for k in T.values.keys():
            d[(k,)] = T.values[k]
        return Tseriesr2(d, T.dim)

#endregion

### Rank 2 augmented Tensor series ###
#region

class Tseriesr2Aug(Tseriesr2):
    def __init__(self, values , dim):
        """
            Parameters
            ----------
            values: dictionary {((int,), int):numpy array}
                the array with key (n_1, ..., n_k) must have shape (d,)**(n_1 + ... + n_k)
            dim: int
                the dimension of the underlying space
        """
        self.values = values
        self.dim = dim
    degree = property(lambda self : max(self._degree(v) for v in self.values.keys()))

    ### Operations on multi-indices

    def _generateKeys(self, N):
        def _gentups(N):
            def conctup(k,l):
                if k==(0,):
                    return l
                if l==(0,):
                    return k
                return k+l
            if N == 0:
                return [(0,)]
            ret = []
            for l in range(1,N+1):
                for k in _gentups(N-l):
                    ret.append(conctup((l,), k))
            return ret

        def _genrecur(N):
            if N == 0:
                return [((0,),)]
            ret = []
            for l in range(1,N+1):
                tups = _gentups(l)
                for k in _genrecur(N-l):
                    if not (type(k[0])==tuple) or (self._isZero(k)):
                        for t in tups:
                            ret.append(self._conc((t,), k))
                    if l==1:
                        ret.append(self._conc((l,), k))
            return ret
        return [i for n in range(N+1) for i in _genrecur(n)]

    def _degree(self, k):
        ret = 0
        for t in k:
            if type(t)==int:
                ret += t
            else:
                ret += sum(t)
        return ret

    def _conc(self, k,l):
        if self._isZero(k):
            return l
        if self._isZero(l):
            return k

        if type(k[-1])==type(l[0]):
            if type(k[-1])==int:
                middle = k[-1]+l[0]
            else:
                middle = super()._conc(k[-1],l[0])
            return k[:-1]+(middle,)+l[1:]
        return k+l

    def _isZero(self, k):
        return k==((0,),) or k==(0,)

    def _isDimZero(self, k):
        return self._isZero(k) or all(type(x)==int for x in k)

    ### Initialisers

    def One():
        return Tseriesr2Aug.fromTens(Tseries.One(), False)

    def Zero():
        return Tseriesr2Aug.fromTens(Tseries.Zero(), False)
    
    def fromVec(v):
        return Tseriesr2Aug.fromTens(Tseries.fromVec(v), False)

    def fromFloat(f):
        return Tseriesr2Aug.fromTens(Tseries.fromFloat(f), False)

    def fromTens(T, Augment = True):
        d = {}
        if Augment:
            d[(1,)] = np.array([1])
        for k in T.values.keys():
            d[((k,),)] = T.values[k]
        return Tseriesr2Aug(d, T.dim)

#endregion