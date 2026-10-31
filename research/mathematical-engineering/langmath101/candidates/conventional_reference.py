"""Independent NumPy forward equations for the three conventional controls.

Reads named weight arrays only; never calls model.forward or PyTorch operators.
This is a numerical reference, not a Lean/IEEE/AD proof.
"""
import math
import numpy as np

def parameters(model):return {k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
def sigmoid(x):return .5*(1+np.tanh(x*.5))
def linear(x,p,name):return x@p[name+'.weight'].T+p[name+'.bias']
def norm(x,p,name):
 z=x-x.mean(-1,keepdims=True)
 return z/np.sqrt((z*z).mean(-1,keepdims=True)+1e-5)*p[name+'.weight']+p[name+'.bias']
def head(x,p):
 x=norm(x,p,'head.norm');g=linear(x,p,'head.gate')
 return linear(linear(x,p,'head.up')*g*sigmoid(g),p,'head.out')
def softmax(x):
 e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def forward(arm,p,tokens,initial=None):
 tokens=np.asarray(tokens,dtype=np.int64);b,n=tokens.shape
 if arm=='gru':
  e=np.tanh(p['embedding.weight'][tokens]);h=np.zeros((b,64),dtype=e.dtype) if initial is None else initial.copy();out=[]
  for t in range(n):
   i=e[:,t]@p['recurrent.weight_ih_l0'].T+p['recurrent.bias_ih_l0']
   s=h@p['recurrent.weight_hh_l0'].T+p['recurrent.bias_hh_l0']
   ir,iz,inn=np.split(i,3,-1);hr,hz,hn=np.split(s,3,-1)
   r=sigmoid(ir+hr);z=sigmoid(iz+hz);off=np.tanh(inn+r*hn)
   h=(1-z)*off+z*h;out.append(h.copy())
  return head(np.stack(out,1),p),{'hidden':h}
 if arm=='selective_ssm':
  e=np.tanh(p['embedding.weight'][tokens]);a=.999*sigmoid(linear(e,p,'retain'));offer=np.tanh(linear(e,p,'offer'))
  s=np.zeros((b,64),dtype=e.dtype) if initial is None else initial.copy();out=[]
  for t in range(n):s=a[:,t]*s+(1-a[:,t])*offer[:,t];out.append(s.copy())
  states=np.stack(out,1);h=np.tanh(linear(states,p,'readout')+linear(e,p,'current'))
  return head(h,p),{'hidden':s}
 if arm=='transformer':
  if initial is not None:tokens=np.concatenate((initial,tokens),axis=1)
  length=tokens.shape[1];e=p['embedding.weight'][tokens]
  positions=np.arange(length,dtype=e.dtype)[:,None];freq=np.exp(np.arange(0,32,2,dtype=e.dtype)*(-math.log(10000.)/32))
  phase=positions*freq;pe=np.stack((np.sin(phase),np.cos(phase)),-1).reshape(length,32)
  h=e*math.sqrt(32)+pe[None];z=norm(h,p,'attention_norm')
  q,k,v=np.split(z@p['attention.in_proj_weight'].T+p['attention.in_proj_bias'],3,-1)
  q,k,v=(x.reshape(b,length,4,8).transpose(0,2,1,3) for x in (q,k,v))
  scores=q@k.transpose(0,1,3,2)/math.sqrt(8)
  scores=np.where(np.triu(np.ones((length,length),dtype=bool),1)[None,None],-np.inf,scores)
  att=(softmax(scores)@v).transpose(0,2,1,3).reshape(b,length,32)
  h=h+linear(att,p,'attention.out_proj');x=linear(norm(h,p,'ffn_norm'),p,'ffn_up')
  gelu=.5*x*(1+np.vectorize(math.erf,otypes=[float])(x/math.sqrt(2)))
  h=h+linear(gelu,p,'ffn_out');return head(h[:,-n:],p),{'tokens':tokens}
 raise ValueError('unknown control')
