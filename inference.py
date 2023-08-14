import torch
from msnet import MSNET
import models.cifar as models
from utils.ms_net_utils import *
from utils.data_utils import *


def inference_with_experts_and_routers(test_loader, experts, router, topk=2):

    """ function to perform evaluation with experts
    params
    -------
    test_loader: data loader for testing dataset
    experts: dictionary of expert Neural Networks
    router: router network
    topK: upto how many top-K you want to re-check?
    """
    freqMat = np.zeros((100, 100)) # -- debug
    router.eval()
    experts_on_stack = []
    expert_count = {} 
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
        expert_count[k] = 0
    
    count = 0
    ext_ = '.png'
    correct = 0
    by_experts, by_router = 0, 0
    mistake_by_experts, mistake_by_router = 0, 0
    agree, disagree = 0, 0

    for dta, target in test_loader:
        count += 1
        dta, target = dta.cuda(), target.cuda()
        output_raw = router(dta)
        output = F.softmax(output_raw)
        router_confs, router_preds = torch.sort(output, dim=1, descending=True)
        preds = []
        confs = []
        for k in range(0, topk):
            #ref = torch.argsort(output, dim=1, descending=True)[0:, k]
            ref = router_preds[0:, k]
            conf = router_confs[0:, k]
            preds.append(ref.detach().cpu().numpy()[0]) # simply put the number. not the graph
            confs.append(conf.detach().cpu().numpy()[0])
    
        cuda0 = torch.device('cuda:0')
        experts_output = []
        router_confident = True
        for exp_ in experts_on_stack:
            if (str(preds[0]) in exp_ and str(preds[1]) in exp_):
                router_confident = False
                break
        
        list_of_experts = []
        target_string = str(target.cpu().numpy()[0])
        for exp in experts_on_stack: #
            if (target_string in exp and (str(preds[0]) in exp or str(preds[1]) in exp)):
                router_confident = False
                list_of_experts.append(exp)
                expert_count[exp] += 1
                #break

        if (router_confident):
            if (preds[0] == target.cpu().numpy()[0]):
                correct += 1
                by_router += 1
            else:
                mistake_by_router += 1
                    
        else:
            for exp in experts_on_stack: #and
                if ( (str(preds[0]) in exp and str(preds[1]) in exp)):                        
                    list_of_experts.append(exp)
                    expert_count[exp] += 1
                    break
           
        #and target_string in str(preds[0]) and target_string in str(preds[1])
        experts_output = [experts[exp_](dta) for exp_ in list_of_experts]
        experts_output.append(output_raw)
        experts_output_avg = average(experts_output)
        experts_output_prob = F.softmax(experts_output_avg, dim=1)
        #pred = torch.argsort(experts_output_prob, dim=1, descending=True)[0:, 0]
        exp_conf, exp_pred = torch.sort(experts_output_prob, dim=1, descending=True)
        pred, conf_ = exp_pred[0:, 0], exp_conf[0:, 0]

        if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
            correct += 1
            by_experts += 1
        else:
            freqMat[pred.cpu().numpy()[0]][target.cpu().numpy()[0]]  += 1
            freqMat[target.cpu().numpy()[0]][pred.cpu().numpy()[0]]  += 1
            mistake_by_experts += 1
        if (pred.cpu().numpy()[0]  == preds[0] \
            and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
            agree += 1
        elif (pred.cpu().numpy()[0]  != preds[0]\
                and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
            disagree += 1
            final_pred, final_conf =  pred.detach().cpu().numpy()[0], conf_.detach().cpu().numpy()[0]
    # commens: The following can be made a function.
            # Save misclassified samples
            args.save_images = False
            if (args.save_images):
                data_numpy = dta[0].cpu() # transfer to the CPU.
                f_name = '%d'%count + '%s'%ext_ # set file name with ext
                f_name_no_text = '%d'%count + 'no_text' + '%s'%ext_
                if (not os.path.exists(args.corrected_images)):
                    os.makedirs(args.corrected_images)
                imshow(data_numpy, os.path.join(args.corrected_images, f_name), \
                    os.path.join(args.corrected_images, f_name_no_text), \
                    fexpertpred=class_rev[final_pred], fexpertconf=final_conf, \
                        frouterpred=class_rev[preds[0]], frouterconf=confs[0])
                
            
    print ("Router and experts agrees with {} samplers \n and router and experts disagres for {}".format(agree, disagree))        
    print ("Routers: {} \n Experts: {}".format(by_router, by_experts))
    print ("Mistakes by Routers: {} \n Mistakes by Experts: {}".format(mistake_by_router, mistake_by_experts))
    print (expert_count)
    print (correct)
    return correct, freqMat, disagree