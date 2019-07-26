import numpy as np

class PQKMEANS:
    def __init__(self,codeword,k,iteration,verbose = True):
        assert len(codeword)>0
        self.codewords_ = codeword
        self.M_ = codeword.shape[0]
        self.K_ = k
        self.iteration_ = iteration
        self.verbose_ = verbose
        Ks = codeword.shape[1]
        self.assignements_ = None
        self.centers_ = None
        assert Ks <= 256
        self.distance_matrices_among_codewords_ = np.zeros((self.M_,Ks,Ks), dtype=int)
        for m in range(self.M_):
            for k1 in range(Ks):
                for k2 in range(Ks):
                    self.distance_matrices_among_codewords_[m][k1][k2] = self.L2SquaredDistance(self.codewords_[m][k1],self.codewords_[m][k2])


    def L2SquaredDistance(self,vec1,vec2):
        assert vec1.shape == vec2.shape
        dist = np.sum((vec1-vec2)**2)
        return dist

    def GetAssignments(self):
        return self.assignements_

    def GetClusterCenters(self):
        return self.centers_

    def SetClusterCenters(self,center_new):
        assert center_new.shape == self.centers_.shape
        self.center_ = center_new

    def SymmetricDistance(self,code1,code2):
        dist = 0
        for m in range(self.M_):
            # print(m,code1[m],code2[m])
            dist+= self.distance_matrices_among_codewords_[m][code1[m]][code2[m]]
        return dist

    def NthCode(self, long_code, n):
        # print("long_code: ",long_code)
        # print("jsbdfkbk",n,self.M_)
        return long_code[n*self.M_:(n+1)*self.M_]

    def NthCodeMthElement(self,long_code,n,m):
        return long_code[n*self.M_+m]

    def InitializeCentersByRandomPicking(self,code, K ):
        centers = np.zeros((K,self.M_), dtype=int)
        ids = np.arange(code.shape[0]//self.M_)
        np.random.shuffle(ids)
        # print("centers********",centers)
        for k in range(K):
            # print(type(centers),type(code),type(ids))
            centers[k] = self.NthCode(code,ids[k])
            # print("center[k]: ",centers[k])
        return centers

    def FindNearetCenterLinear(self,query,codes):
        dists = np.zeros((codes.shape[0]), dtype=int)
        sz = codes.shape[0]
        for i in range(sz): # can be parallelised
            # print(codes[i][0])
            dists[i] = self.SymmetricDistance(query,codes[i])
        min_dist = float("inf")
        min_i = -1
        for i in range(sz):
            if dists[i] < min_dist:
                min_i = i
                min_dist = dists[i]
        assert min_i != -1
        return min_i, min_dist

    def ComputeCenterBySparseVoting(self,codes, selected_ids):
        average_code = np.zeros((self.M_), dtype=int)
        Ks = self.codewords_.shape[1]
        for m in range(self.M_):
            frequency_histogram = np.zeros((Ks), dtype=int)
            for id in selected_ids:
                frequency_histogram[self.NthCodeMthElement(codes,id,m)]+=1
            vote = np.zeros((Ks), dtype=int)
            for k1 in range(Ks):
                freq = frequency_histogram[k1]
                if freq == 0:
                    continue
                for k2 in range(Ks):
                    vote[k2] += freq * self.distance_matrices_among_codewords_[m][k1][k2]
            min_dist = float("inf")
            min_ks = -1
            for ks in range(Ks):
                if vote[ks] < min_dist:
                    min_ks = ks
                    min_dist = vote[ks]
            assert min_ks != -1
            average_code[m] = min_ks
        return average_code

    def fit(self,pydata):
        print("pydata.shape",pydata.shape)
        assert self.K_ * self.M_ <= pydata.shape[0]
        assert pydata.shape[0] % self.M_ == 0

        N = pydata.shape[0]//self.M_
        print("N:",N,"self.M_",self.M_)
        self.center_ = np.zeros((self.K_,self.M_), dtype=int)
        self.assignements_ = np.zeros((N), dtype=int)
        centers_new = self.InitializeCentersByRandomPicking(pydata,self.K_)
        errors = np.zeros((N), dtype=int)
        for itr in range(self.iteration_):
            print("centersnew, random: ", centers_new)            
            if self.verbose_:
                print("iteration: ", itr)
            centers_old = centers_new
            error_sum = 0
            selected_indices_foreach_center = [[] for i in range(self.K_)]
            for n in range(N):
                # print("center old: ",centers_old)
                min_k_dist = self.FindNearetCenterLinear(self.NthCode(pydata,n),centers_old)
                self.assignements_[n],errors[n] = min_k_dist
            for n in range(N):
                k = self.assignements_[n]
                selected_indices_foreach_center[k].append(n)
                error_sum+=errors[n]
            if itr != self.iteration_ -1:
                for k in range(self.K_):
                    if len(selected_indices_foreach_center[k]) ==0:
                        continue
                    centers_new[k] = self.ComputeCenterBySparseVoting(pydata,selected_indices_foreach_center[k])
        self.center_ = centers_new

    def predict_one(self,pyvector):
        assert pyvector.shape == self.M_.shape
        nearest_one = self.FindNearetCenterLinear(pyvector,self.center_)
        return nearest_one[0]
            


            
