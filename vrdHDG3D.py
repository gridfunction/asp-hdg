# Solves tau u - Lap u = f on [0,1]^3
# ngsolve-imports
from netgen.csg import *
from ngsolve.krylovspace import CGSolver
from ngsolve.meshes import *
from ngsolve import *
from numpy import pi
# petsc solver
import ngs_petsc 
import time as timeit

ngsglobals.msg_level=0
SetHeapSize(int(1e9))
import sys
if len(sys.argv) > 1:
    order =   int(sys.argv[1])
    refines = int(sys.argv[2])
    freeBC = int(sys.argv[3])>0
else:
    order = 1
    refines = 4
    freeBC = False
print("Order = ", order, "Free = ", freeBC)
# diffusion/reaction parameters
D = 1+ (x-0.5)**2+(y-0.5)**2 +(z-0.5)**2
tau =  1 + sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)

def SolveProblem(order=order, refines=refines,
        condense=True,symmetric=False):
    for i in range(refines):
        t0 = timeit.time()
        mesh = MakeStructured3DMesh(hexes=False, nx=8*2**i, ny = 8*2**i, 
                nz = 8*2**i)
        t1 = timeit.time()
        print("\nElasped:%.2e MESHING "%(t1-t0))
        
        # Div-HDG spaces
        V = HDiv(mesh, order=order, dirichlet=".*")
        if freeBC ==True:
            M = TangentialFacetFESpace(mesh, order=order, 
                highest_order_dc=True)
            # aux H1 space
            V0 = VectorH1(mesh, order=1, 
                    dirichlety="left|right", dirichletx="front|back", 
                    dirichletz="top|bottom")
        else: # dir bc
            M = TangentialFacetFESpace(mesh, order=order, 
                highest_order_dc=True, dirichlet=".*")
            # aux H1 space
            V0 = VectorH1(mesh, order=1, dirichlet=".*")

        fes = FESpace([V,M])
        gfu = GridFunction (fes)
        (u, uhat), (v, vhat) = fes.TnT()
        (u0, v0) = V0.TnT()

        # gradients
        gradv, gradu = Grad(v), Grad(u)
        
        # RHS (constant) 
        f = LinearForm (fes)
        f += (v[0]+v[1]+v[2])*dx
        
        # normal direction and mesh size
        n = specialcf.normal(mesh.dim)
        h = specialcf.mesh_size
        # stability parameter
        #alpha = 8*order**2/h
        alpha = 8*order**2/h
        
        # tangential component
        def tang(v):
            return v-(v*n)*n
        
        ########### HDG operator ah
        a = BilinearForm (fes, symmetric=True, condense=True)
        # volume term
        a += (D*InnerProduct(gradu,gradv)+tau*u*v)*dx
        # bdry terms
        a += D*(-gradu*n*tang(v-vhat)-gradv*n*tang(u-uhat)
                +alpha*tang(u-uhat)*tang(v-vhat))*dx(element_boundary=True)
        
        ######## face path blocks for smoother
        def facePatchBlocks(mesh, fes):
            blocks = []
            freedofs = fes.FreeDofs(True)
            for e in mesh.faces:
                edofs = set()
                # get ALL dofs connected to the face
                for el in mesh[e].elements:
                    edofs |= set(d for d in fes.GetDofNrs(el)
                                 if freedofs[d])
                blocks.append(edofs)
            return blocks
        
        def faceBlocks(mesh, fes):
            blocks = []
            freedofs = fes.FreeDofs(True)
            for e in mesh.faces:
                edofs = set(d for d in fes.GetDofNrs(e)
                             if freedofs[d])
                blocks.append(edofs)
            return blocks
        
        ######## THIS IS MOST TIME CONSUMING PART
        fBlocks = facePatchBlocks(mesh, fes)
        fBlocks0 = faceBlocks(mesh, fes)
        t0 = timeit.time()
        # number of DOFS
        ntotal = fes.ndof
        nglobal = sum(fes.FreeDofs(True))
        print("Elasped:%.2e BLOCKING  Total DOFs: %.2e Global DOFs: %.2e "%(
            t0-t1, ntotal, nglobal))
        
        class SymmetricGS(BaseMatrix):
              def __init__ (self, smoother):
                  super(SymmetricGS, self).__init__()
                  self.smoother = smoother
              def Mult (self, x, y):
                  y[:] = 0.0
                  self.smoother.Smooth(y, x)
                  self.smoother.SmoothBack(y,x)
              def Height (self):
                  return self.smoother.height
              def Width (self):
                  return self.smoother.height
        
        ########### ASP operator ah0
        a0 = BilinearForm(V0, symmetric=True)
        a0 += (D*InnerProduct(Grad(u0), Grad(v0))+tau*u0*v0)*dx
        
        # Projection operator V0--> fes
        # We set up a mixed mass matrix for V0 -> fes and then solving with the mass matrix in M
        mixmass = BilinearForm(trialspace=V0, testspace=fes)
        # tangential part
        mixmass += tang(u0) * tang(vhat) * dx(element_boundary=True)
        # normal part
        mixmass += (u0*n) * (v*n) * dx(element_boundary=True)
        
        massf = BilinearForm(fes)
        massf += tang(uhat) * tang(vhat) * dx(element_boundary=True)
        massf += (u*n) * (v*n) * dx(element_boundary=True)

        with TaskManager():
            f.Assemble()
            a.Assemble()
            t1 = timeit.time()
            print("Elasped:%.2e ASSEMBLE "%(t1-t0))
            ######### smoother (use edge blocks)
            # smoother
            jac = a.mat.CreateSmoother(fes.FreeDofs(True))
            bjac = a.mat.CreateBlockSmoother(fBlocks)

            ######## ASP
            a0.Assemble()
            pm = ngs_petsc.PETScMatrix(a0.mat, V0.FreeDofs())
            inva0 = ngs_petsc.PETSc2NGsPrecond(pm, 
                    petsc_options = {"pc_type": "hypre"})
            massf.Assemble()
            mixmass.Assemble()
            # massf is block-diagonal in 3D!!!!!!!!!
            m_inv = massf.mat.CreateBlockSmoother(fBlocks0) 

            E = m_inv @ mixmass.mat
            ET = mixmass.mat.T @ m_inv
            pre_twogrid = E @ inva0 @ ET

            # jacobi smoother
            pre1 = jac + pre_twogrid
            # block gs smoother
            pre2 = SymmetricGS(bjac) + pre_twogrid
            t2 = timeit.time()
            print("Elasped:%.2e PREC "%(t2-t1))

            inv1 = CGSolver(a.mat, pre1, maxsteps=10000, 
                precision=1e-10, printrates=True)
            inv2 = CGSolver(a.mat, pre2, maxsteps=10000, 
                precision=1e-10, printrates=True)
            
            f.vec.data += a.harmonic_extension_trans * f.vec
            # solver
            gfu.vec.data = inv1*f.vec
            gfu.vec.data = inv2*f.vec
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
            t3 = timeit.time()
            print("Elasped:%.2e SOLVE "%(t3-t2))
            print("JAC:%i  BGS: %i"%(inv1.GetSteps(), inv2.GetSteps()))
            
            print("*****************************************************")
            print("*****************************************************")
            print("*****************************************************")

SolveProblem(order, refines)
