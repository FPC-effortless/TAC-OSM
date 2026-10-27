"""Cross-check the COMPLETE language graph, historical head, losses and AD."""
import unittest
import torch
import torch.nn.functional as F
from parallel_tac_language_gate_original import ParallelTACLanguage
from source_independent_reference import run,historical

class FullGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self,mode,pos):
        torch.manual_seed(442)
        return ParallelTACLanguage(decoder=75,post=10,gate_mode=mode,position_mode=pos).double().eval()

    def test_all_typed_forward_states_and_historical_reads(self):
        for mode in ("tanh","silu"):
            for position in ("legacy","no_time","multiscale","multiscale_relative"):
                for length in (1,2,5,9):
                    model=self.model(mode,position)
                    token=torch.randint(128,(2,length),generator=torch.Generator().manual_seed(700+length))
                    output,trace=model(token,return_trace=True)
                    reference,last,before=run(model,token,trace=True)
                    self.assertLess((output-reference).detach().abs().max().item(),5e-11,(mode,position,length))
                    self.assertLess((trace["aligned"][:,-1]-last["aligned"]).detach().abs().max().item(),5e-11)
                    if length>1:
                        self.assertLess((trace["aligned"][:,:-1]-before[:,1:]).detach().abs().max().item(),5e-11)
                        lag=min(2,length-1)
                        observed,target=model.historical_logits(token,trace,lag)
                        predicted,expected=historical(model,token,before,lag)
                        self.assertTrue(torch.equal(target,expected))
                        self.assertLess((observed-predicted).detach().abs().max().item(),5e-11)

    def test_all_parameter_groups_next_token_and_historical_ad(self):
        for mode in ("tanh","silu"):
            for pos in ("legacy","multiscale_relative"):
                model=self.model(mode,pos)
                token=torch.randint(128,(2,8),generator=torch.Generator().manual_seed(5))
                target=torch.randint(128,(2,8),generator=torch.Generator().manual_seed(6))
                output,trace=model(token,return_trace=True)
                hist,hist_target=model.historical_logits(token,trace,2)
                reference,_,before=run(model,token,trace=True)
                hist_ref,hist_target_ref=historical(model,token,before,2)
                source_loss=F.cross_entropy(output.reshape(-1,128),target.flatten())+\
                    .3*F.cross_entropy(hist.reshape(-1,128),hist_target.flatten())
                ref_loss=F.cross_entropy(reference.reshape(-1,128),target.flatten())+\
                    .3*F.cross_entropy(hist_ref.reshape(-1,128),hist_target_ref.flatten())
                self.assertLess(abs((source_loss-ref_loss).detach().item()),1e-10)
                named=list(model.named_parameters())
                source_grad=torch.autograd.grad(source_loss,[p for _,p in named],allow_unused=True)
                ref_grad=torch.autograd.grad(ref_loss,[p for _,p in named],allow_unused=True)
                active=0
                for (name,_),a,b in zip(named,source_grad,ref_grad):
                    self.assertEqual(a is None,b is None,name)
                    if a is not None:
                        self.assertTrue(torch.isfinite(a).all(),name)
                        self.assertLess((a-b).detach().abs().max().item(),2e-9,name)
                        active+=1
                self.assertGreater(active,20)

    def test_actual_64_character_historical_query_graph(self):
        # The registered research objective uses t-64, not merely short lags.
        model=self.model("silu","multiscale_relative")
        token=torch.randint(128,(1,65),generator=torch.Generator().manual_seed(99))
        output,trace=model(token,return_trace=True)
        expected,final,before=run(model,token,trace=True)
        self.assertLess((output-expected).detach().abs().max().item(),1e-10)
        found,target=model.historical_logits(token,trace,64)
        ref,ref_target=historical(model,token,before,64)
        self.assertTrue(torch.equal(target,ref_target))
        self.assertLess((found-ref).detach().abs().max().item(),1e-10)

    def test_prefix_future_isolation(self):
        model=self.model("silu","multiscale_relative")
        token=torch.randint(128,(2,6))
        suffix=torch.randint(128,(2,4))
        a,_,past=run(model,token,trace=True)
        b,_,longpast=run(model,torch.cat((token,suffix),1),trace=True)
        self.assertTrue(torch.equal(a,b[:,:6]))
        self.assertTrue(torch.equal(past,longpast[:,:6]))

    def test_scan_fourblock_vs_shared_rectangular_full_backprop(self):
        model=self.model("silu","multiscale_relative")
        token=torch.randint(128,(2,7))
        model.optimized_scan=True
        a=model(token).square().sum()
        model.optimized_scan=False
        b=model(token).square().sum()
        self.assertLess(abs((a-b).detach().item()),1e-9)
        named=list(model.named_parameters())
        ga=torch.autograd.grad(a,[p for _,p in named],allow_unused=True)
        gb=torch.autograd.grad(b,[p for _,p in named],allow_unused=True)
        for (name,_),x,y in zip(named,ga,gb):
            self.assertEqual(x is None,y is None,name)
            if x is not None:
                self.assertLess((x-y).detach().abs().max().item(),1e-8,name)
if __name__=="__main__":
    unittest.main()
