"""Independent real-valued slope references for the TAC-LANG activation paths.

Diagnostic only. Uses local, constructed values rather than training or
claims about authentic language performance. Derivatives are independently
computed from scalar calculus and compared against Torch AD and two-sided
finite differences on source-shaped operations.
"""
import unittest
import torch
import torch.nn.functional as F

class LanguageNonlinearDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_silu_source_write_gate_all_saturation_regimes(self):
        for x in (-12.,-4.,-1.,-0.1,0.,0.1,1.,4.,12.):
            for base in (-7.,0.,7.):
                r=torch.tensor(x,dtype=torch.float64,requires_grad=True)
                b=torch.tensor(base,dtype=torch.float64)
                inner=2*r
                silu=inner*torch.sigmoid(inner)
                gate=torch.sigmoid(b+silu)
                grad=torch.autograd.grad(gate,r)[0]
                sig_inner=float(torch.sigmoid(inner).detach())
                g=float(gate.detach())
                closed=g*(1-g)*2*(sig_inner+2*x*sig_inner*(1-sig_inner))
                self.assertAlmostEqual(float(grad),closed,delta=1e-11)
                eps=1e-5
                with torch.no_grad():
                    def val(z):
                        v=2*z
                        return float(torch.sigmoid(b+v*torch.sigmoid(v)))
                    fd=(val(r+eps)-val(r-eps))/(2*eps)
                self.assertAlmostEqual(fd,closed,delta=5e-7)

    def test_linear_source_write_gate_has_distinct_gradient(self):
        for x in (-9.,0.,9.):
            r=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            gate=torch.sigmoid(r)
            grad=torch.autograd.grad(gate,r)[0]
            slope=float(gate.detach())*(1-float(gate.detach()))
            self.assertAlmostEqual(float(grad),slope,delta=1e-12)

    def test_complete_248_layer_norm_directional_calculus(self):
        rng=torch.Generator().manual_seed(128883)
        for scale in (0.,1e-6,0.1,1.,10.):
            z=(torch.randn(248,generator=rng,dtype=torch.float64)*scale)
            z.requires_grad_(True)
            direction=torch.randn(248,generator=rng,dtype=torch.float64)
            direction=direction/direction.norm()
            gamma=torch.randn(248,generator=rng,dtype=torch.float64)
            bias=torch.randn(248,generator=rng,dtype=torch.float64)
            upstream=torch.randn(248,generator=rng,dtype=torch.float64)
            eps=1e-5
            objective=(F.layer_norm(z,(248,),gamma,bias,eps)*upstream).sum()
            backward=torch.autograd.grad(objective,z)[0]
            mu=z.detach().mean()
            xc=z.detach()-mu
            var=(xc*xc).mean()
            dmu=direction.mean()
            dvar=2*(xc*(direction-dmu)).mean()
            denom=torch.sqrt(var+eps)
            dnorm=(direction-dmu)/denom-xc*dvar/(2*denom**3)
            closed=(upstream*gamma*dnorm).sum()
            predicted=(backward*direction).sum()
            self.assertAlmostEqual(float(predicted),float(closed),delta=1e-8)
            step=5e-6
            with torch.no_grad():
                def objective_at(v):
                    return (F.layer_norm(v,(248,),gamma,bias,eps)*upstream).sum()
                observed=(objective_at(z+step*direction)-
                          objective_at(z-step*direction))/(2*step)
            # Central differences carry O(h^2) truncation amplified by small
            # sqrt(variance+eps); retain strict Torch-vs-analytic check.
            self.assertAlmostEqual(float(observed),float(closed),
                                   delta=5e-7+1e-8*abs(float(closed)))

if __name__=="__main__":
    unittest.main()
