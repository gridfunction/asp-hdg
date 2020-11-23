# Solves - Lap u + tau u = f on [0,1]^2
# ngsolve-imports
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
block1 = x1*y1 + x2*y2

tau =  tau0 + block1*(tau1-tau0)

def SolveProblem(order=order, refines=refines):
    for i in range(refines):
        t0 = timeit.time()
        mesh = MakeStructured2DMesh(quads=False, nx=8*2**i, ny = 8*2**i)
        t1 = timeit.time()
        #print("Elasped:%.2e MESHING "%(t1-t0))
        
        # Div-HDG spaces
        V = HDiv(mesh, order=order, dirichlet=".*")
        M = TangentialFacetFESpace(mesh, order=order, 
                highest_order_dc=True, dirichlet=".*")
        fes = FESpace([V,M])
        # aux H1 space
        V0 = VectorH1(mesh, order=1, dirichlet=".*")
        
        gfu = GridFunction (fes)
        (u,uhat),  (v,vhat) = fes.TnT()
        u0, v0 = V0.TnT()

        # gradient by row
        gradv, gradu = Grad(v), Grad(u)
        
        # RHS (constant) 
        f = LinearForm (fes)
        f += (v[0]+v[1])*dx

        # normal direction and mesh size
        n = specialcf.normal(mesh.dim)
        h = specialcf.mesh_size
        # stability parameter
        alpha = 4*order**2/h
        
        # tangential component
        def tang(v):
            return v-(v*n)*n

        ########### HDG operator ah
        a = BilinearForm (fes, symmetric=True, condense=True)
        # volume term
        a += (InnerProduct(gradu,gradv)+tau*u*v)*dx
        # bdry terms
        a += (-gradu*n*tang(v-vhat)-gradv*n*tang(u-uhat)
                +alpha*tang(u-uhat)*tang(v-vhat))*dx(element_boundary=True)
        

        
        ######## facet patch (edge path) for block smoother
        def edgePatchBlocks(mesh, fes):
            blocks = []
            freedofs = fes.FreeDofs(True)
            for e in mesh.edges:
                edofs = set()
                # get ALL dofs connected to the edge
                for el in mesh[e].elements:
                    edofs |= set(d for d in fes.GetDofNrs(el)
                                 if freedofs[d])
                blocks.append(edofs)
            return blocks
        
        eBlocks = edgePatchBlocks(mesh, fes)
        t0 = timeit.time()
        #print("Elasped:%.2e BLOCKING "%(t0-t1))
        
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
        a0 = BilinearForm(V0)
        a0 += (InnerProduct(Grad(u0), Grad(v0))+tau*u0*v0)*dx
        
        
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
            #print("Elasped:%.2e ASSEMBLE "%(t1-t0))
            ######### smoother (use edge blocks)
            jac = a.mat.CreateSmoother(fes.FreeDofs(True))
            bjac = a.mat.CreateBlockSmoother(eBlocks)


            ######## ASP
            a0.Assemble()
            pm = ngs_petsc.PETScMatrix(a0.mat, V0.FreeDofs())
            inva1 = ngs_petsc.PETSc2NGsPrecond(pm, 
                    petsc_options = {"pc_type": "hypre"})
            
            massf.Assemble()
            mixmass.Assemble()
            # massf is diagonal!!!!!!!!!
            m_inv = massf.mat.CreateSmoother(fes.FreeDofs(True)) 

            E = m_inv @ mixmass.mat
            ET = mixmass.mat.T @ m_inv
            pre_twogrid = E @ inva1 @ ET

            # jacobi smoother
            pre1 = jac + pre_twogrid
            # block gs smoother
            pre2 = SymmetricGS(bjac) + pre_twogrid
            t2 = timeit.time()
            #print("Elasped:%.2e PREC "%(t2-t1))
            
            
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
            #print("Elasped:%.2e SOLVE \n  tau0:%f tau1 %f   JAC:%i  BGS: %i"%(t3-t2, 
            # tau0, tau1,  inv1.GetSteps(), inv2.GetSteps()))
            print("tau0:%.0e tau1 %.0e   JAC:%i  BGS: %i"%(tau0, tau1,  inv1.GetSteps(), inv2.GetSteps()))
            
            ndof = fes.ndof
            ndof_g = M.ndof
            #print('\n', ndof, ndof-ndof_g, ndof_g)

SolveProblem(order, refines)
