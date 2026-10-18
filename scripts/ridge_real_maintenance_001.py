"""Predeclared deterministic linear autoregression, selected on validation only."""
import torch


def fit_ridge(train,validation):
    mean=train['x'].mean((0,1)).double();scale=train['x'].std((0,1)).clamp_min(.01).double();target=(train['y'].double()-mean)/scale;records=[];best=None
    for lags in (1,24):
        def design(x):
            z=((x[:,-lags:].double()-mean)/scale).flatten(1)
            return torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1)
        x=design(train['x']);v=design(validation['x']);cov=x.T@x;rhs=x.T@target
        for penalty in (.01,1.,100.):
            regularizer=torch.eye(x.shape[1],dtype=x.dtype)*penalty;regularizer[-1,-1]=0;weights=torch.linalg.solve(cov+regularizer,rhs);error=float(((v@weights-(validation['y'].double()-mean)/scale)**2).mean());record={'lags':lags,'penalty':penalty,'validation_normalized_MSE':error};records.append(record)
            if best is None or (error,lags,penalty)<best[0]:best=((error,lags,penalty),{'mean':mean,'scale':scale,'weights':weights,'lags':lags,'penalty':penalty})
    return best[1],records


def predict_ridge(model,x):
    z=((x[:,-model['lags']:].double()-model['mean'])/model['scale']).flatten(1);return torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1)@model['weights']
