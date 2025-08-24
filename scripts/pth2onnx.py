import torch
import os
import sys
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models import cifar as models


class PthToOnnxConverter:
    def __init__(self, model, dummy_input, export_path='./output/',
                 input_names=['input'], output_names=['output'], dynamic_axes=None):
        self.model = model
        self.dummy_input = dummy_input
        self.export_path = export_path
        os.makedirs(self.export_path, exist_ok=True)

        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes or {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    def __call__(self, pth_path):
        try:
            checkpoint = torch.load(pth_path, map_location='cpu')

            # Handle common checkpoint formats
            if 'net' in checkpoint:
                state_dict = checkpoint['net']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return

        export_model_name = os.path.basename(pth_path).replace('.pth', '.onnx').replace('.tar', '.onnx')
        export_file = os.path.join(self.export_path, export_model_name)

        try:
            torch.onnx.export(
                self.model,
                self.dummy_input,
                export_file,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                opset_version=11,  # use modern opset
            )
            print(f"✅ Model has been converted to ONNX and saved at {export_file}")

        except Exception as e:
            print(f"❌ Error exporting to ONNX: {e}")


if __name__ == "__main__":
    pth_path = '/Users/intisar/Documents/research/source/msnets/DeepExpertEnsembles/workspace/cifar100/c100_5.5/checkpoint_experts'
    pth_paths = [os.path.join(pth_path, f) for f in os.listdir(pth_path) if f.endswith(('.pth', '.tar'))]

    # Build your model
    model = models.__dict__['resnet'](num_classes=100, depth=20, block_name='BasicBlock')

    # Converter
    converter = PthToOnnxConverter(
        model=model,
        dummy_input=torch.randn(1, 3, 32, 32),
        export_path='./onnx_exports/'
    )

    # Convert all checkpoints
    for pth in pth_paths:
        converter(pth)
