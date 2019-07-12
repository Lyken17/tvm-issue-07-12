#### Waht the code do

There are two models, the two models are modified from ProxylessNAS GPU (https://github.com/mit-han-lab/ProxylessNAS).

Both models are torch.Module. For each model:

 1. Transform the torch model to onnx format
 2. Transform the onnx format model to relay model
 3. Compile the relay model
 4. Run the compiled model on GPU
 
The second model can run successfully. But the first model will fail at the four step.

The first model contains the 2~18 blocks in the original Proxyless Mobile model and the second model contains the 2~17 blocks in the original Proxyless Mobile model.
(I changed the out_channels of the first conv layer and in_channels of the mix_feature layer in Proxyless Mobile model to fit the above change of blocks)

#### Reproduce

run 
```python
$ python demo.py
```
Then my machine, the output is:
```text
Testing proxyless_net1
Exception occurs: 
Traceback (most recent call last):
  [bt] (3) /home/yaoyao/miniconda3/envs/python37/lib/python3.7/site-packages/tvm-0.6.dev0-py3.7-linux-x86_64.egg/tvm/libtvm.so(TVMFuncCall+0x65) [0x7f5f53b04ff5]
  [bt] (2) /home/yaoyao/miniconda3/envs/python37/lib/python3.7/site-packages/tvm-0.6.dev0-py3.7-linux-x86_64.egg/tvm/libtvm.so(std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::detail::PackFuncVoidAddr_<4, tvm::runtime::CUDAWrappedFunc>(tvm::runtime::CUDAWrappedFunc, std::vector<tvm::runtime::detail::ArgConvertCode, std::allocator<tvm::runtime::detail::ArgConvertCode> > const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)+0xb6) [0x7f5f53b81fb6]
  [bt] (1) /home/yaoyao/miniconda3/envs/python37/lib/python3.7/site-packages/tvm-0.6.dev0-py3.7-linux-x86_64.egg/tvm/libtvm.so(tvm::runtime::CUDAWrappedFunc::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*, void**) const+0x832) [0x7f5f53b81e32]
  [bt] (0) /home/yaoyao/miniconda3/envs/python37/lib/python3.7/site-packages/tvm-0.6.dev0-py3.7-linux-x86_64.egg/tvm/libtvm.so(dmlc::LogMessageFatal::~LogMessageFatal()+0x43) [0x7f5f533c2a33]
  File "/home/yaoyao/repos/tvm/src/runtime/cuda/cuda_module.cc", line 111
  File "/home/yaoyao/repos/tvm/src/runtime/module_util.cc", line 73
CUDAError: Check failed: ret == 0 (-1 vs. 0) : cuModuleLoadData(&(module_[device_id]), data_.c_str()) failed with error: CUDA_ERROR_INVALID_PTX
Testing proxyless_net2
Successfully.
```




