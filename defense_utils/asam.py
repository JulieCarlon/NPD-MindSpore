import mindspore as ms
from mindspore import Tensor, Parameter, ops
import mindspore.common.dtype as mstype

## https://zhuanlan.zhihu.com/p/595716023

@ms.jit_class
class ASAM():
    def __init__(self, optimizer,rho):
        self.optimizer = optimizer
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.map = ops.HyperMap()
        self.old_param = optimizer.parameters.clone(prefix="para_", init='zeros')
        self.rho = rho

    def __call__(self, grads,first_step=True):
        
        if first_step:
            params = self.optimizer.parameters   
            self.map(ops.partial(ops.assign), self.old_param, params) # record the old parameters
            scales = []
            for i in range(len(params)):
                scale = ops.norm(ops.mul(ops.abs(params[i]),grads[i]),2)
                if scale is not None:
                    scales.append(scale)
            scales = ops.norm(ops.stack(scale),2)
            for i in range(len(grads)):
                temp = self.rho*ops.mul(ops.pow(params[i],2),grads[i])/(scales+1e-12)
                ops.assign_add(self.inner_grads[i], temp)
            self.optimizer(self.inner_grads)
         
        else:
            self.map(ops.partial(ops.assign), self.optimizer.parameters, self.old_param) # recover the old parameters
            self.optimizer(grads)
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)

        return True