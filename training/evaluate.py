from nff.train import evaluate
import torch
from torch.autograd import Variable


def evaluate_model(model, normalizer, model_type, dataloader, loss_fn, gpu_num):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)

    if model_type == "Painn":
        return evaluate(model, dataloader, loss_fn, device=gpu_num)

    model.eval()
    loss_cumulative = 0.   
    loss_direct_cumulative = 0.
    loss_contrastive_cumulative = 0.
    contrastive_term_count = 0.
    predictions = []
    targets = []
    total_count = 0
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            if model_type == "CGCNN":
                input_struct = d[0]
                target = d[1]
                input_var = (Variable(input_struct[0].cuda(non_blocking=True)),
                             Variable(input_struct[1].cuda(non_blocking=True)),
                             input_struct[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input_struct[3]])
                output = model(*input_var).view(-1)
                target = Variable(target.cuda(non_blocking=True))
            else:
                d.to(device)
                output = model(d)
                target = d.target
                
            prediction = normalizer.denorm(output)
            predictions.append(prediction)
            targets.append(target)
            if model_type == "CGCNN":
                loss = loss_fn(normalizer.denorm(output), target)
                loss_cumulative = loss_cumulative + loss.detach().item()*target.shape[0]
            elif model_type == "e3nn_contrastive":
                loss, direct_loss, contrastive_loss = loss_fn(normalizer.denorm(output), d.target, d.comp)
                loss_cumulative = loss_cumulative + loss.detach().item()
                loss_direct_cumulative = loss_direct_cumulative + direct_loss.detach().item()*target.shape[0]
                curr_contrastive_count = count_contastive_terms(d.comp)
                loss_contrastive_cumulative = loss_contrastive_cumulative + contrastive_loss.detach().item()*curr_contrastive_count
                contrastive_term_count += curr_contrastive_count
            else:
                loss = loss_fn(normalizer.denorm(output), d.target)
                loss_cumulative = loss_cumulative + loss.detach().item()*target.shape[0]
            
            total_count += target.shape[0]
        
        if model_type == "e3nn_contrastive":
            loss_output = [loss_cumulative/len(dataloader),loss_direct_cumulative/total_count,loss_contrastive_cumulative/contrastive_term_count]
        else:
            loss_output = [loss_cumulative/total_count]
    
    return torch.cat(predictions), torch.cat(targets), loss_output



def count_contastive_terms(comps):
    last_index = 0
    ordering_count = 0
    stored_comp = comps[0]
    for i in range(len(comps)):
        curr_comp = comps[i]
        if curr_comp != stored_comp:
            length = i-last_index
            ordering_count += (length)**2 - length
            last_index = i
            stored_comp = comps[i]

    length = len(comps)-last_index
    ordering_count += (length)**2 - length

    return ordering_count