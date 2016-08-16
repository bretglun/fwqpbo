import ctypes
import numpy as np
import sys
import os

IMGTYPE = ctypes.c_float
gyro=42.58

# Configure the QPBO graphcut function from the c++ DLL
def init_QPBOcpp():
    DLLdir = os.path.dirname(__file__)
    if '32 bit' in sys.version: DLLfile = 'QPBO32'
    else: DLLfile = 'QPBO64'
    try: lib = np.ctypeslib.load_library(DLLfile, DLLdir)
    except:
        print(sys.exc_info())
        raise Exception('{}.dll not found in dir "{}"'.format(DLLfile,DLLdir))
    QPBOcpp = lib[1] # Does not work to access the function by name: lib.fwqpbo
    QPBOcpp .restype = None # Needed for void functions
    
    QPBOcpp.argtypes = [ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int, 
                        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(ctypes.c_int, flags='aligned, contiguous')]
    return QPBOcpp

#TODO: implement Fibonacci search
def greedyR2(J,nVxl):
    nR2 = J.shape[0]
    R2 = np.zeros(shape=(nVxl))
    for i in range(nVxl):
        r = 0
        min = J[r,i]
        if r+1<nR2 and J[r+1,i]<min:
            while r+1<nR2 and J[r+1,i]<min:
                min = J[r+1,i]
                r+=1
        R2[i]=r
    return R2

def getR2Residuals(Y,dB0,CBt,nB0,nR2,nVxl):
    J = np.zeros(shape=(nR2,nVxl))
    for b in range(nB0): 
        for r in range(nR2): 
            J[r,dB0==b]=np.linalg.norm(np.dot(CBt[b][r],Y[:,dB0==b]),axis=0)**2
    return J
    
def ICM(previous,L,maxICMUpdate,nICMiter,J,V,wx,wy,wz,left,right,up,down,above,below):
    current = np.array(previous)
    for k in range(nICMiter): # ICM iterate
        print(str(k+1),', ',end='')
        previous[:] = current[:]
        min_cost = np.full(current.shape,np.inf)        

        updates=[0]*(2*maxICMUpdate+1) # Update order
        updates[2:len(updates):2]=list(range(1,maxICMUpdate+1)) # Even are positive
        updates[1:len(updates):2]=list(range(-1,-maxICMUpdate-1,-1)) # Odd are negative
        for update in updates:
            cost = J[(previous+update)%L,range(J.shape[1])] # Unary cost
            # Binary costs:
            cost[right]+=wx*V[abs((previous[right]+update)%L-previous[left])] # left cost
            cost[left]+=wx*V[abs((previous[left]+update)%L-previous[right])] # right cost
            cost[down]+=wy*V[abs((previous[down]+update)%L-previous[up])] # up cost
            cost[up]+=wy*V[abs((previous[up]+update)%L-previous[down])] # down cost
            cost[below]+=wz*V[abs((previous[below]+update)%L-previous[above])] # above cost
            cost[above]+=wz*V[abs((previous[above]+update)%L-previous[below])] # below cost
            
            current[cost<min_cost]=(previous[cost<min_cost]+update)%L
            min_cost[cost<min_cost]=cost[cost<min_cost]
    return current

# Find all local minima of discretely evaluated function f(t) with period T
def findLocalMinima(f): return np.where((f<np.roll(f,1))*(f<np.roll(f,-1)))[0]

# In each voxel, find the two smallest local residual minima in a period of omega
def findTwoSmallestMinima(J):
    nVxl = J.shape[1]
    A = np.zeros(nVxl,dtype=int)
    B = np.zeros(nVxl,dtype=int)
    for i in range(nVxl):
        minima = sorted(findLocalMinima(J[:,i]),key=lambda x: J[x,i])[:2]
        if len(minima)==2: A[i],B[i] = minima
        elif len(minima)==1: A[i]=B[i]=minima[0]
        else: A[i]=B[i]=0 # Assign dummy minimum
    return A,B

def getIndexImages(nx,ny,nz):
    left = np.zeros((nz,ny,nx),dtype=bool)
    left[:,:,:-1]=True
    right = np.zeros((nz,ny,nx),dtype=bool)
    right[:,:,1:]=True
    down = np.zeros((nz,ny,nx),dtype=bool)
    down[:,:-1,:]=True
    up = np.zeros((nz,ny,nx),dtype=bool)
    up[:,1:,:]=True
    below = np.zeros((nz,ny,nx),dtype=bool)
    below[:-1,:,:]=True
    above = np.zeros((nz,ny,nx),dtype=bool)
    above[1:,:,:]=True
    return left.flatten(),right.flatten(),up.flatten(),down.flatten(),above.flatten(),below.flatten()

# 2D measure of isotropy defined as the square area over the square perimeter (area normalized to 1)
def isotropy2D(dx,dy): return np.sqrt(dx*dy)/(2*(dx+dy))

# 3D measure of isotropy defined as the cube volume over the cube area (volume normalized to 1)
def isotropy3D(dx,dy,dz): return (dx*dy*dz)**(2/3)/(2*(dx*dy+dx*dz+dy*dz))

def getHigherLevel(level):
    highLevel = {'L':level['L']+1}
    # Isotropy promoting downsampling
    maxIsotropy = 0
    for sx in [1,2]:
        for sy in [1,2]:
            for sz in [1,2]: # Loop over all 2^3=8 downscaling combinations
                if sx*sy*sz>1 and level['nx']>=sx and level['ny']>=sy and level['nz']>=sz: # at least one dimension must change and the size of all dimensions at lower level must permit any downscaling                
                    if (level['nx']==1): iso = isotropy2D(level['dy']*sy,level['dz']*sz)
                    elif (level['ny']==1): iso = isotropy2D(level['dx']*sx,level['dz']*sz)
                    elif (level['nz']==1): iso = isotropy2D(level['dx']*sx,level['dy']*sy)
                    else: iso = isotropy3D(level['dx']*sx,level['dy']*sy,level['dz']*sz)
                    if iso>maxIsotropy:
                        maxIsotropy=iso
                        highLevel['sx']=sx
                        highLevel['sy']=sy
                        highLevel['sz']=sz
    highLevel['dx']=level['dx']*highLevel['sx']
    highLevel['dy']=level['dy']*highLevel['sy']
    highLevel['dz']=level['dz']*highLevel['sz']
    
    highLevel['nx']=int(np.ceil(level['nx']/highLevel['sx']))
    highLevel['ny']=int(np.ceil(level['ny']/highLevel['sy']))
    highLevel['nz']=int(np.ceil(level['nz']/highLevel['sz']))
    return highLevel

def getHighLevelResidualImage(J,highLevel,level):
    Jlow = np.zeros((J.shape[0],level['nz']+level['nz']%highLevel['sz'],level['ny']+level['ny']%highLevel['sy'],level['nx']+level['nx']%highLevel['sx']))
    Jlow[:,:level['nz'],:level['ny'],:level['nx']] = J.reshape(J.shape[0],level['nz'],level['ny'],level['nx'])
    
    Jhigh = np.zeros((J.shape[0],highLevel['nz'],highLevel['ny'],highLevel['nx']))
    
    Jhigh = Jlow[:,::highLevel['sz'],::highLevel['sy'],::highLevel['sx']]
    if highLevel['sx']>1: Jhigh += Jlow[:,::highLevel['sz'],::highLevel['sy'],1::highLevel['sx']]
    if highLevel['sy']>1: Jhigh += Jlow[:,::highLevel['sz'],1::highLevel['sy'],::highLevel['sx']]
    if highLevel['sz']>1: Jhigh += Jlow[:,1::highLevel['sz'],::highLevel['sy'],::highLevel['sx']]
    if highLevel['sx']>1 and highLevel['sy']>1: Jhigh += Jlow[:,::highLevel['sz'],1::highLevel['sy'],1::highLevel['sx']]
    if highLevel['sx']>1 and highLevel['sz']>1: Jhigh += Jlow[:,1::highLevel['sz'],::highLevel['sy'],1::highLevel['sx']]
    if highLevel['sy']>1 and highLevel['sz']>1: Jhigh += Jlow[:,1::highLevel['sz'],1::highLevel['sy'],::highLevel['sx']]
    if highLevel['sx']>1 and highLevel['sy']>1 and highLevel['sz']>1: Jhigh += Jlow[:,1::highLevel['sz'],1::highLevel['sy'],1::highLevel['sx']]
    
    return Jhigh.reshape(Jhigh.shape[0],-1)/(highLevel['sx']*highLevel['sy']*highLevel['sz']) # Scale

def getB0fromHighLevel(dB0high,level,highLevel):
    dB0 = np.empty((highLevel['nz']*highLevel['sz'],highLevel['ny']*highLevel['sy'],highLevel['nx']*highLevel['sx']),dtype=int)
    dB0[::highLevel['sz'],::highLevel['sy'],::highLevel['sx']] = dB0high
    if highLevel['sx']>1: dB0[::highLevel['sz'],::highLevel['sy'],1::highLevel['sx']] = dB0high
    if highLevel['sy']>1: dB0[::highLevel['sz'],1::highLevel['sy'],::highLevel['sx']] = dB0high
    if highLevel['sz']>1: dB0[1::highLevel['sz'],::highLevel['sy'],::highLevel['sx']] = dB0high
    if highLevel['sx']>1 and highLevel['sy']>1: dB0[::highLevel['sz'],1::highLevel['sy'],1::highLevel['sx']] = dB0high
    if highLevel['sx']>1 and highLevel['sz']>1: dB0[1::highLevel['sz'],::highLevel['sy'],1::highLevel['sx']] = dB0high
    if highLevel['sy']>1 and highLevel['sz']>1: dB0[1::highLevel['sz'],1::highLevel['sy'],::highLevel['sx']] = dB0high
    if highLevel['sx']>1 and highLevel['sy']>1 and highLevel['sz']>1: dB0[1::highLevel['sz'],1::highLevel['sy'],1::highLevel['sx']] = dB0high
    return dB0[:level['nz'],:level['ny'],:level['nx']].flatten()

def calculateFieldMap(nB0,level,graphcutLevel,multiScale,maxICMupdate,nICMiter,J,V,mu):
    A,B = findTwoSmallestMinima(J)
    dB0 = np.array(A)
    
    ### Multiscale recursion
    if dB0.shape[0]==1: # Trivial case at coarsest level with only one voxel
        print('Level (1,1,1): Trivial case')
        return dB0

    if multiScale:
        highLevel = getHigherLevel(level)
        Jhigh = getHighLevelResidualImage(J,highLevel,level)
        # Recursion:
        dB0high = calculateFieldMap(nB0,highLevel,graphcutLevel,multiScale,maxICMupdate,nICMiter,Jhigh,V,mu).reshape(highLevel['nz'],highLevel['ny'],highLevel['nx']) 
        dB0 = getB0fromHighLevel(dB0high,level,highLevel)
        print('Level ({},{},{}): '.format(level['nx'],level['ny'],level['nz']))
    
    ### Prepare MRF
    print('Preparing MRF...',end='')
    
    # Prepare data fidelity costs
    D = np.array([J[A,range(J.shape[1])],J[B,range(J.shape[1])]],dtype=IMGTYPE)
    
    # Prepare discontinuity costs
    vxls = range(J.shape[1])
    # 2nd derivative of residual function
    # NOTE: No division by square(steplength) since square(steplength) not included in V
    ddJ = (J[(A+1)%nB0,vxls]+J[(A-1)%nB0,vxls]-2*J[A,vxls])
                                                
    left,right,up,down,above,below = getIndexImages(level['nx'],level['ny'],level['nz'])

    wx = np.minimum(ddJ[left],ddJ[right])*mu/level['dx']
    wy = np.minimum(ddJ[down],ddJ[up])*mu/level['dy']
    wz = np.minimum(ddJ[below],ddJ[above])*mu/level['dz']
    print('DONE')
    
    ### QPBO    
    graphcut = level['L']>=graphcutLevel
    if graphcut:        
        Vx = np.array(wx*[V[abs(A[left]-A[right])],V[abs(A[left]-B[right])],V[abs(B[left]-A[right])],V[abs(B[left]-B[right])]],dtype=IMGTYPE)
        Vy = np.array(wy*[V[abs(A[down]-A[up])],V[abs(A[down]-B[up])],V[abs(B[down]-A[up])],V[abs(B[down]-B[up])]],dtype=IMGTYPE)
        Vz = np.array(wz*[V[abs(A[below]-A[above])],V[abs(A[below]-B[above])],V[abs(B[below]-A[above])],V[abs(B[below]-B[above])]],dtype=IMGTYPE)
        
        label = np.zeros(ddJ.shape,dtype=ctypes.c_int)
        
        print('Solving MRF using QPBO...',end='')
        QPBOcpp = init_QPBOcpp() # Initialize c++ function
        QPBOcpp(level['nx'],level['ny'],level['nz'],D,Vx,Vy,Vz,label)
        print('DONE')
    
        dB0[label==0] = A[label==0]
        dB0[label==1] = B[label==1]

    ### ICM
    if nICMiter>0:
        print('Solving MRF using ICM...',end='')
        dB0 = ICM(dB0,nB0,maxICMupdate,nICMiter,J,V,wx,wy,wz,left,right,up,down,above,below)    
        print('DONE')
    return dB0

def getB0Residuals(Y,CBt,nB0,nVxl,iR2cand):
    J = np.zeros(shape=(nB0,nVxl))
    # TODO: loop over all R2candidates
    for b in range(nB0): J[b,:]=np.linalg.norm(np.dot(CBt[b][iR2cand[0]],Y),axis=0)**2
    return J

# Construct demodulation vectors for each B0 value
def demodulationVectors(nB0,N):
    Bt = []
    for b in range(nB0):
        omega = 2.*np.pi*b/nB0
        Bt.append(np.eye(N)+0j*np.eye(N))
        for n in range(N): Bt[b][n,n] = np.exp(complex(0.,-n*omega))
    return Bt

# Construct modelMatrix M
def modelMatrix(dPar,mPar,R2):
    RA = np.zeros(shape=(dPar.N,mPar.M))+1j*np.zeros(shape=(dPar.N,mPar.M))
    for n in range(dPar.N):
        t = dPar.t1+n*dPar.dt
        RA[n,0] = np.exp(complex(-t*R2,0)) # Water resonance
        for p in range(1,mPar.P): # Loop over fat resonances
            omega = 2.*np.pi*gyro*dPar.B0*(mPar.CS[p]-mPar.CS[0]) # Chemical shift between water and peak m (in ppm)
            RA[n,1]+=mPar.alpha[1][p]*np.exp(complex(-t*R2,t*omega))
    return RA

# Perform the actual reconstruction
def reconstruct(dPar,aPar,mPar,B0map=None,R2map=None):
    determineB0 = aPar.graphcutLevel<20 or aPar.nICMiter>0
    nR2 = aPar.nR2
    determineR2 = nR2>1
    if (nR2<0): nR2 = -nR2 # nR2<(-1) will use input R2map
    
    nVxl = dPar.nx*dPar.ny*dPar.nz
    
    Y = dPar.img
    Y.shape = (dPar.N,nVxl)
    
    # Prepare matrices
    RA,RAp,C = [],[],[]
    for r in range(nR2):
        R2 = r*aPar.R2step;
        RA.append(modelMatrix(dPar,mPar,R2))
        RAp.append(np.linalg.pinv(RA[r]))
        C.append(np.eye(dPar.N)-np.dot(RA[r],RAp[r])) # Null space projection matrices
    
    Bt = demodulationVectors(aPar.nB0,dPar.N)	# Off-resonance demodulation vectors (one for each off-resonance value)
    CBt = [] # Product of C and Bt
    for b in range(aPar.nB0):
        CBt.append([])
        for r in range(nR2): CBt[b].append(np.dot(C[r],Bt[b]))
    
    if determineB0:
        V = [] #Precalculate discontinuity costs
        for b in range(aPar.nB0): V.append(min(b**2,(b-aPar.nB0)**2))
        V = np.array(V)
        
        level = {'L':0,'nx':dPar.nx,'ny':dPar.ny,'nz':dPar.nz,'sx':1,'sy':1,'sz':1,'dx':dPar.dx,'dy':dPar.dy,'dz':dPar.dz}
        J = getB0Residuals(Y,CBt,aPar.nB0,nVxl,aPar.iR2cand)
        dB0 = calculateFieldMap(aPar.nB0,level,aPar.graphcutLevel,aPar.multiScale,aPar.maxICMupdate,aPar.nICMiter,J,V,aPar.mu)
    
    if determineR2:    
        J = getR2Residuals(Y,dB0,CBt,aPar.nB0,nR2,nVxl)
        R2 = greedyR2(J,nVxl)
        
    # Find least squares solution given dB0 and R2
    rho=np.zeros(shape=(mPar.M,nVxl))+1j*np.zeros(shape=(mPar.M,nVxl))
    for r in range(nR2): 
        for b in range(aPar.nB0):
            rho[:,(dB0==b)*(R2==r)] = np.dot(np.dot(RAp[r],Bt[b]),Y[:,(dB0==b)*(R2==r)])
    
    if B0map is None: B0map = np.empty(nVxl,dtype=IMGTYPE)
    if R2map is None: R2map = np.empty(nVxl,dtype=IMGTYPE)
    
    if determineR2: R2map[:]=R2*aPar.R2step
    
    if determineB0:
        B0step = 1.0/aPar.nB0/dPar.dt/gyro/dPar.B0 # For B0 index -> off-resonance in ppm
        # //if (iB0>nB0/2) iB0 = iB0-nB0; // "Wrap" off-resonance period about zero
        B0map[:]=dB0*B0step
    
    return rho,B0map,R2map