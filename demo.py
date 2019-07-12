import io
import logging
import numpy as np
import tvm
import tvm.relay as relay
import tvm.contrib.graph_runtime as runtime
import torch
import torch.onnx
import onnx
from proxyless_nas import proxyless_net

logging.disable(logging.WARNING)


def from_torch(module, dummy_inputs):
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs = (dummy_inputs,)
    input_shape = {}
    for index, dummy_input in enumerate(dummy_inputs):
        if isinstance(dummy_input, np.ndarray):
            dummy_input = torch.from_numpy(dummy_input)
        input_shape[str(index)] = dummy_input.shape

    buffer = io.BytesIO()
    module.eval()
    torch.onnx.export(module, dummy_inputs, buffer)
    buffer.seek(0, 0)
    onnx_model = onnx.load_model(buffer)
    return tvm.relay.frontend.from_onnx(onnx_model, shape=input_shape)


def torch_inference_on_gpu(torch_module, torch_inputs):
    relay_module, params = from_torch(torch_module, torch_inputs)
    print("Build TVM relay")
    with tvm.relay.build_config(opt_level=1):
        graph, lib, params = relay.build(relay_module, target='cuda', params=params)
    print("Finish TVM relay")
    ctx = tvm.gpu(1)
    tvm_module = runtime.create(graph, lib, ctx)

    tvm_inputs = {'0': tvm.nd.array(torch_inputs.detach().numpy())}
    tvm_module.set_input(**tvm_inputs)
    tvm_module.set_input(**params)

    tvm_module.run()
    tvm_outputs = tvm_module.get_output(0, tvm.nd.empty((1, 1000)))

    return tvm_outputs


def test_all():
    from torchvision.models import resnet50
    nets = [
        ('resnet50', resnet50(),),
        ('proxyless_net1', proxyless_net(1)),
        ('proxyless_net2', proxyless_net(2))
    ]
    torch_input = torch.rand(1, 3, 224, 224)
    for model_name, net in nets:
        print(f"Testing {model_name}")
        try:
            tvm_result = torch_inference_on_gpu(net, torch_input)
            torch_result = net(torch_input)
            tvm.testing.assert_allclose(tvm_result.asnumpy(), torch_result.detach().numpy(), atol=1e-2, rtol=1e-2)
        except Exception as e:
            print(f"Exception occurs: \n{e}")
            exit(0)
        else:
            print(f"Successfully.")


test_all()
