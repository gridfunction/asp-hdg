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

import sys
if len(sys.argv) > 1:
    order =   int(sys.argv[1])
    refines = int(sys.argv[2])
    tau0 = float(sys.argv[3])
    tau1 = float(sys.argv[4])
else:
    order = 1
    refines = 4
    tau0 = 1
    tau1 = 1

# reaction parameter
x1 = IfPos((x-0.5)*(x-0.25), 0,1)
x2 = IfPos((x-0.5)*(x-0.75), 0,1)
y1 = IfPos((y-0.5)*(y-0.25), 0,1)
y2 = IfPos((y-0.5)*(y-0.75), 0,1)
z1 = IfPos((z-0.5)*(z-0.25), 0,1)
z2 = IfPos((z-0.5)*(z-0.75), 0,1)
block1 = x1*y1*z1 + x2*y2*z2


tau =  tau0 + block1*(tau1-tau0)

def SolveProblem(order=order, refines=refines,
        condense=True,symmetric=False):
    for i in range(refines):
        t0 = timeit.time()
        mesh = MakeStructured3DMesh(hexes=False, nx=8*2**i, ny = 8*2**i, 
                nz = 8*2**i)
        t1 = timeit.time()
        print("\nElasped:%.2e MESHING "%(t1-t0))
        
        # HDG spaces 
        V = L2(mesh, order=order)
        M = FacetFESpace(mesh, order=order, highest_order_dc=True, 
                dirichlet=".*")
        fes = FESpace([V,M])
        # aux-H1 space
        V0 = H1(mesh, order=1, dirichlet=".*")
        
        gfu = GridFunction (fes)
        (u, uhat), (v, vhat) = fes.TnT()
        (u0, v0) = V0.TnT()

        # gradients
        gradv, gradu = grad(v), grad(u)
        
        # RHS (constant) 
        f = LinearForm (fes)
        f += v*dx
        
        # normal direction and mesh size
        n = specialcf.normal(mesh.dim)
        h = specialcf.mesh_size
        # stability parameter
        alpha = 4*order**2/h
        
        ########### HDG operator ah
        a = BilinearForm (fes, symmetric=True, condense=True)
        # volume term
        a += (gradu*gradv+tau*u*v)*dx
        # bdry terms
        a += (-gradu*n*(v-vhat)-gradv*n*(u-uhat)
                +alpha*(u-uhat)*(v-vhat))*dx(element_boundary=True)
        
        
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
        
        ######## THIS IS MOST TIME CONSUMING PART
        fBlocks = facePatchBlocks(mesh, fes)
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
        a0 += (grad(u0)*grad(v0)+tau*u0*v0)*dx
        
        # Projection operator H1--> M
        # We set up a mixed mass matrix
        # for H1 -> M and then solving with the mass matrix in M
        mixmass = BilinearForm(trialspace=V0, testspace=fes)
        mixmass += u0 * vhat * dx(element_boundary=True)
        
        uhat1, vhat1 = M.TnT()
        massf = BilinearForm(M, symmetric=True)
        massf += uhat1 * vhat1 * dx(element_boundary=True)

        embM = Embedding(fes.ndof, fes.Range(1))
        
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
            # massf is diagonal
            m_inv0 = massf.mat.CreateSmoother(M.FreeDofs(True)) 
            m_inv = embM @ m_inv0 @ embM.T

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
                precision=1e-10, printrates=False)
            inv2 = CGSolver(a.mat, pre2, maxsteps=10000, 
                precision=1e-10, printrates=False)
            
            f.vec.data += a.harmonic_extension_trans * f.vec
            # solver
            gfu.vec.data = inv1*f.vec
            gfu.vec.data = inv2*f.vec
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
            t3 = timeit.time()
            print("Elasped:%.2e SOLVE "%(t3-t2))
            print("tau0:%.0e tau1 %.0e   JAC:%i  BGS: %i"%(tau0, tau1,  inv1.GetSteps(), inv2.GetSteps()))
            
            print("*****************************************************")
            print("*****************************************************")
            print("*****************************************************")

SolveProblem(order, refines)
