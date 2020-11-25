# Solves - Lap^2 phi + tau Lap phi = f on [0,1]^2
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
    bc = str(sys.argv[5])
else:
    order = 1
    refines = 4
    tau0 = 1
    tau1 = 1
    bc = "simple"


# reaction parameter
x1 = IfPos((x-0.5)*(x-0.25), 0,1)
x2 = IfPos((x-0.5)*(x-0.75), 0,1)
y1 = IfPos((y-0.5)*(y-0.25), 0,1)
y2 = IfPos((y-0.5)*(y-0.75), 0,1)
block1 = x1*y1 + x2*y2

tau =  tau0 + block1*(tau1-tau0)

def hesse(v):
    return v.Operator("hesse")


def SolveProblem(order=order, refines=refines):
    # load mesh from file
    for i in range(refines):
        t0 = timeit.time()
        mesh = MakeStructured2DMesh(quads=False, nx=8*2**i, ny = 8*2**i)
        t1 = timeit.time()
        
        # C0-HDG FESpaces
        VT = H1(mesh, order=order+1, dirichlet=".*")
        if bc == "clamped": # clamped bc
           VF = TangentialFacetFESpace(mesh, order=order-1, 
                   dirichlet=".*")# with bdry condition
        else:
            VF = TangentialFacetFESpace(mesh, order=order-1) # no bdry condition
        V = FESpace([VT, VF])

        gfu = GridFunction (V)
        (xi, uhat), (zeta,vhat) = V.TnT()
        
        n = specialcf.normal(2)
        h = specialcf.mesh_size
        def tang(v):
            return v-(v*n)*n
        
        u = CoefficientFunction((grad(xi)[1], -grad(xi)[0]))
        v = CoefficientFunction((grad(zeta)[1], -grad(zeta)[0]))
        hess_zeta = zeta.Operator("hesse")
        # gradient by row 
        gradv = CoefficientFunction(( (hess_zeta[1], hess_zeta[3]),
            (-hess_zeta[0], -hess_zeta[1])), 
            dims =(2,2))
        hess_xi = xi.Operator("hesse")
        gradu = CoefficientFunction((
            (hess_xi[1], hess_xi[3]),
            (-hess_xi[0], -hess_xi[1])), 
            dims =(2,2))
        
        f = LinearForm (V)
        f += zeta*dx 
        
        a = BilinearForm (V, condense=True, symmetric=True)
        lap = tau*u*v
        bilap = InnerProduct(gradu, gradv)
        bilap_BND = -(gradu*n*tang(v-vhat)+gradv*n*tang(u-uhat)
                -4*(order+1)**2/h*tang(u-uhat)*tang(v-vhat))
        
        ir = IntegrationRule(SEGM, 2*order-1)
        
        a += (lap+bilap)*dx
        a += bilap_BND*dx(element_boundary=True, intrules={SEGM:ir})


        ########### ASP 1: C0--> HDiv
        ########### ASP 1: C0--> HDiv
        ########### ASP 1: C0--> HDiv
        # two grid (Hdiv-HDG)
        VT1 = HDiv(mesh, order=order, hodivfree=True,dirichlet='.*')
        V1 = FESpace([VT1, VF])
        (u1, uhat1), (v1, vhat1) = V1.TnT()
        
        # Mapping HDiv--> H1 
        mixmass = BilinearForm(trialspace=V1, testspace=V)
        # tangential part
        mixmass += u1*v * dx
        
        # construct Pi-hat 
        xiT, zetaT = VT.TnT()
        uT = CoefficientFunction((grad(xiT)[1], -grad(xiT)[0]))
        vT = CoefficientFunction((grad(zetaT)[1], -grad(zetaT)[0]))
        uhatT, vhatT = VF.TnT()
        
        # tangential part
        massfX = BilinearForm(VT, symmetric=True)
        massfX += uT*vT* dx

        massfY = BilinearForm(trialspace=VT, testspace=VF)
        massfY += -tang(uT)*tang(vhatT)*dx(element_boundary=True)

        massfZ = BilinearForm(VF)
        massfZ += tang(uhatT)*tang(vhatT)* dx(element_boundary=True)
        
        ############### DOFS splits for PROJECTOR 
        embU = Embedding(V.ndof, V.Range(0))
        embV = Embedding(V.ndof, V.Range(1))
        
        a1 = BilinearForm(V1, condense=True)
        lap1 = tau*u1*v1
        gradu1 = Grad(u1)
        gradv1 = Grad(v1)
        bilap1 = InnerProduct(gradu1, gradv1)
        
        jmp_u1 = tang(u1-uhat1)
        jmp_v1 = tang(v1-vhat1)
        bilap_BND1 = -(gradu1*n*jmp_v1+gradv1*n*jmp_u1
                -4*(order+1)**2/h*jmp_u1*jmp_v1)
        a1 += (lap1+bilap1)*dx
        a1 += bilap_BND1*dx(element_boundary=True, intrules={SEGM:ir})

        ########### ASP 2: HDiv--> H1
        ########### FIXME: weakly impose bdry condition
        if bc == "clamped":
          V0 = VectorH1(mesh,dirichlet=".*")
        else:
          V0 = VectorH1(mesh,dirichletx="left|right", dirichlety="top|bottom")
        u0, v0 = V0.TnT()
        a0 = BilinearForm(V0)
        a0 += (InnerProduct(Grad(u0), Grad(v0))+tau*u0*v0)*dx
        
        # Projection operator H1--> M
        mixmass0 = BilinearForm(trialspace=V0, testspace=V1)
        # tangential part
        mixmass0 += tang(u0) * tang(vhat1) * dx(element_boundary=True)
        mixmass0 += (u0*n) * (v1*n) * dx(element_boundary=True)
        
        massf0 = BilinearForm(V1)
        # tangential part
        massf0 += tang(uhat1) * tang(vhat1) * dx(element_boundary=True)
        massf0 += (u1*n) * (v1*n) * dx(element_boundary=True)
        
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
        
        eBlocks = edgePatchBlocks(mesh, V1)
        
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


        with TaskManager():
            f.Assemble()
            a.Assemble()
            ######## ASP 1 
            mixmass.Assemble()
            massfX.Assemble()
            massfY.Assemble()
            massfZ.Assemble()
            ####### RK: PROJECTION (sparse cholesky factorization)
            imX = massfX.mat.Inverse(VT.FreeDofs(True),
                inverse="sparsecholesky")
            
            imZ = massfZ.mat.CreateSmoother(VF.FreeDofs(True))
            m_inv = embU @ imX @ embU.T + embV @ imZ @embV.T \
                    - embV @ imZ @ massfY.mat @ imX @ embU.T
            m_invT = embU @ imX @ embU.T + embV @ imZ @embV.T \
                    - embU @ imX @ massfY.mat.T @ imZ @ embV.T

            E = m_inv @ mixmass.mat
            ET = mixmass.mat.T @ m_invT

            a1.Assemble()
            ## ASP 2
            bjac = a1.mat.CreateBlockSmoother(eBlocks)
            bgs = SymmetricGS(bjac)

            mixmass0.Assemble()
            massf0.Assemble()
            m_inv0 = massf0.mat.CreateSmoother(V1.FreeDofs(True)) 
            
            a0.Assemble()
            pm0 = ngs_petsc.PETScMatrix(a0.mat, V0.FreeDofs())
            inva0 = ngs_petsc.PETSc2NGsPrecond(pm0, 
                    petsc_options = {"pc_type": "hypre"})
            E0 = m_inv0 @ mixmass0.mat
            # ASP for rd system
            pc_a1 = bgs + E0 @ inva0 @ E0.T

            # Direct solver for rd system
            pc_a2 = a1.mat.Inverse(V1.FreeDofs(True),
                inverse="sparsecholesky")

            pre1 = E @ pc_a1 @ ET
            pre2 = E @ pc_a2 @ ET

            inv1 = CGSolver(a.mat, pre1, maxsteps=5000, precision=1e-10, printrates=True)
            inv2 = CGSolver(a.mat, pre2, maxsteps=5000, precision=1e-10, printrates=True)
            
            f.vec.data += a.harmonic_extension_trans * f.vec
            # solver
            gfu.vec.data = inv1*f.vec
            gfu.vec.data = inv2*f.vec
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
            print("bc:%s tau0:%.0e tau1 %.0e   ASP:%i  EXA: %i"%(bc, 
              tau0, tau1,  inv1.GetSteps(), inv2.GetSteps()))

SolveProblem(order, refines)
print("\n\n")
