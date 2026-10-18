"""Optimize the information-matched reference, never neural deployment code.

Exact depth2 expected cost: sum_o min_b mass(o)*[C(b)+terminal(q_o T_b)].
This algebra removes Python branch recursion without changing planning semantics.
"""
import argparse
import time
import numpy as np
from data_belief_dev062 import PriceBelief,STATES,COMMANDS
from run_learned_stack_dev055 import write


class VectorReference:
    def __init__(self,b):
        self.transitions=b.transitions.copy();self.cost=np.array([0,b.inspection+.02,b.inspection+.02,.63,.63]);self.terminal=20*(self.transitions@STATES.sum(-1));self.likelihood=np.zeros((5,4,4));self.likelihood[0,0]=1
        for a,(kind,target) in enumerate(COMMANDS[1:],1):
            if kind=='inspect':
                self.likelihood[a,0]=STATES[:,target]==0;self.likelihood[a,1]=STATES[:,target]==1
            else:
                for o,observed in enumerate(STATES):self.likelihood[a,o]=np.prod(np.where(STATES==observed,1-b.error,b.error),axis=-1)

    def costs(self,p):
        after=np.einsum('s,ast->at',p,self.transitions);joint=after[:,None,:]*self.likelihood;mass=joint.sum(-1);continuation=(mass[:,:,None]*(self.cost+.01)+joint@self.terminal.T).min(-1).sum(-1);result=self.cost+continuation;result[0]=.01+20*np.dot(p,STATES.sum(-1));return result

    def policy(self,p):
        v=self.costs(p)
        if p[0]<1-1e-12:v[0]=1e6
        best=v<=v.min()+1e-10;return best/best.sum()


def run(output):
    rng=np.random.default_rng(6269999);checks=0;maxdiff=0.;times=[]
    for parents in ([-1,-1],[-1,0],[1,-1]):
        for error in (.1,.2):
            for price in (.05,.1,.2,.4):
                b=PriceBelief(parents,error,inspection=price);fast=VectorReference(b);probabilities=np.vstack((np.eye(4),rng.dirichlet(np.ones(4),size=48)))
                for p in probabilities:
                    actual=fast.costs(p);expected=b.action_costs(p,2);np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10);b.p=p;np.testing.assert_array_equal(fast.policy(p),b.policy());maxdiff=max(maxdiff,float(np.max(np.abs(actual-expected))));checks+=1
                for label,function in [('recursive',lambda: b.action_costs(probabilities[5],2)),('vectorized',lambda:fast.costs(probabilities[5]))]:
                    samples=[]
                    for _ in range(100):
                        start=time.process_time_ns();function();samples.append((time.process_time_ns()-start)/1e6)
                    times.append({'parents':parents,'error':error,'inspection':price,'variant':label,'median_cpu_ms':float(np.median(samples))})
    write(output,{'status':'COMPLETE','mathematical_equivalence_checks':checks,'maximum_absolute_cost_drift':maxdiff,'measurements':times,'scope':'Same exact known-law PUBLIC depth2 reference and STOP constraint, optimized only for fair compute comparison; no deployed neural component, real repair or ROI claim. Timings exclude filter update, public-record parsing, construction and transport.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
