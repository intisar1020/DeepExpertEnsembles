import os
import torch
import torch.nn as nn
import torch.optim as optim
import models.cifar as models


class MSNET(nn.Module):
    def __init__(self, expert_base=None, named_nodes=None, pretrained=True, ckpt_path=""):
        super().__init__()
        
        if named_nodes is None:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")
            named_nodes = [os.path.splitext(f)[0] for f in os.listdir(ckpt_path) if f.endswith('.pth')]
            
        if not named_nodes:
            raise ValueError("No named nodes provided or found.")
            
        self.named_nodes = named_nodes
        self.node_map = {name: i for i, name in enumerate(self.named_nodes)}
        self.expert_pool = nn.ModuleList([expert_base for _ in range(len(self.named_nodes))])

        if pretrained:
            print(f"Loading pre-trained experts from '{ckpt_path}'...")
            for name in self.named_nodes:
                try:
                    ckpt_file = os.path.join(ckpt_path, f'{name}.pth')
                    wts = torch.load(ckpt_file)
                    self.expert_pool[self.node_map[name]].load_state_dict(wts['net'])
                    print(f"  - Loaded expert: '{name}'")
                except FileNotFoundError:
                    print(f"  - Warning: Checkpoint for '{name}' not found. Skipping.")

    def _ensemble(self, outputs):
        return sum(outputs)

    def forward(self, input_tensor, experts_to_use=None):
        if experts_to_use is None:
            experts_to_use = self.named_nodes
        
        invalid_experts = [name for name in experts_to_use if name not in self.node_map]
        if invalid_experts:
            raise ValueError(f"Invalid expert names provided: {', '.join(invalid_experts)}")

        outputs = [self.expert_pool[self.node_map[name]](input_tensor) for name in experts_to_use]
        return self._ensemble(outputs)


def main():
    CKPT_PATH = 'work_space/all_superset/checkpoint_experts'
    
    base_net = models.__dict__['resnet'](
        num_classes=100,
        depth=20,
        block_name='BasicBlock'
    )
    
    try:
        net = MSNET(expert_base=base_net, ckpt_path=CKPT_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    input_tensor = torch.rand((3, 3, 32, 32))
    
    out_tensor = net(input_tensor)
    print(f"Output shape: {out_tensor.shape}")

if __name__ == "__main__":
    main()