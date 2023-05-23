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
    results = []
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
                output = model(*input_var)
                target = Variable(target.cuda(non_blocking=True))
            else:
                d.to(device)
                output = model(d)
                target = d.target
                
            result = normalizer.denorm(output)
            results.append(result)
            targets.append(target)
            if model_type == "CGCNN":
                loss = loss_fn(normalizer.denorm(output), target)
            elif model_type == "e3nn_contrastive":
                loss = loss_fn(normalizer.denorm(output), d.target, d.comp)
            else:
                loss = loss_fn(normalizer.denorm(output), d.target)
            loss_cumulative = loss_cumulative + loss.detach().item()*target.shape[0]
            total_count += target.shape[0]
    
    return torch.cat(results), torch.cat(targets), loss_cumulative/total_count