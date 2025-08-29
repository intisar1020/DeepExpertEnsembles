from calflops import calculate_flops
# from torchvision import models
import models.cifar as models


model = models.__dict__['resnet'](
    num_classes=100,
    depth=20,
    block_name='BasicBlock')
# model = models.resnet50()

# model = models.__dict__['densenet'](
#     num_classes=100,
#     depth=100,
#     growthRate=12,
# #    compressionRate=,
# )

model = model.cuda()

batch_size = 1
input_shape = (batch_size, 3, 32, 32)
flops, macs, params = calculate_flops(model=model, input_shape=input_shape,
                                      output_as_string=False)
print (flops/10e8, macs, params/10e5)