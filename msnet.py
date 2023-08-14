from operator import index
import os
import torch
import torch.nn as nn
import torch.optim as optim
import models.cifar as models


class MSNET(nn.Module):
    def __init__(self, expert_base=None, router_base=None, named_nodes=['a', 'b', 'c'], pretrained=True, ckpt_path="") -> None:
        """_summary_

        Args:
            expert_base (_type_, optional): _description_. Defaults to None.
            router_base (_type_, optional): _description_. Defaults to None.
            named_nodes (list, optional): _description_. Defaults to ['a', 'b', 'c'].
            pretrained (bool, optional): _description_. Defaults to True.
        """

        super().__init__()
        split_f = lambda x: x.split(".")[0]
        if (named_nodes is None):
            print (os.listdir(ckpt_path))
            named_nodes = [split_f(named_node) for named_node in os.listdir(ckpt_path)]
        self.named_nodes = named_nodes
        self.num_of_nodes = len(self.named_nodes)
        # map named nodes to number for easy access.
        self.en = {str(elem):i for i, elem in enumerate(self.named_nodes)} # en = enumerated_nodes
        self.expert_pool = nn.ModuleList([expert_base for i in range(self.num_of_nodes)])
        if (pretrained):
            for i, ckpt_name in enumerate(self.named_nodes):
                wts = torch.load(os.path.join(ckpt_path, ckpt_name+'.pth'))
                self.expert_pool[self.en[str(ckpt_name)]].load_state_dict(wts['net'])
                print ("Loaded")

    def _ensemble(self, outputs):
        return sum(outputs) #/ len(outputs)    
    
    def forward(self, input_, index_list=list()):
        """ infer through either single experts or ensemble of experts

        Args:
            input_ (_type_): _description_
            index_list (_type_, optional): _description_. Defaults to list().

        Returns:
            _type_: _description_
        """
        if (not len(index_list)):
            index_list = self.named_nodes
       
        output_ = [self.expert_pool[self.en[str(index_)]](input_) for index_ in index_list]
        ensembled_output = self._ensemble(output_)
        return ensembled_output
    




if __name__ == "__main__":
    base_net = models.__dict__['resnet'](
        num_classes=100,
        depth=20,
        block_name='BasicBlock')
    demo_loi = [str(i) for i in range(5)]
    net = MSNET(expert_base=base_net, router_base=base_net, named_nodes=None, ckpt_path='work_space/all_superset/checkpoint_experts')
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4)
    state_dict = net.state_dict()
    index_list = os.listdir('work_space/all_superset/checkpoint_experts')
    split_f = lambda x: x.split(".")[0]
    index_list = [split_f(index_) for index_ in index_list]
    input_ = torch.rand((3, 3, 32, 32))
    out_ = net(input_)
    print (out_.shape)
    #torch.save(state_dict, "demo.pth")
    #print (net.expert_pool[0])
