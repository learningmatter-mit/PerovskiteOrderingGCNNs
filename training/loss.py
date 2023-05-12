import torch

kb = 0.00008617333
T = 1300

def pairwise_probs(output, target):
    ##### Get probabilities
    output_ref = output - torch.min(output)
    target_ref = target - torch.min(target)
    output_prob = torch.exp(-output_ref/(kb*T))
    target_prob = torch.exp(-target_ref/(kb*T))
    output_norm = output_prob / output_prob.sum()
    target_norm = target_prob / target_prob.sum()
    #### Get matrices
    output_clone = output_norm.clone()
    target_clone = target_norm.clone()
    size = target.shape[0]
    
    target_norm = target_norm.view(-1,1).expand(size,size)
    target_clone = target_clone.view(1,-1).expand(size,size)
    
    output_norm = output_norm.view(-1,1).expand(size,size)
    output_clone = output_clone.view(1,-1).expand(size,size)
    
    target_matrix = target_norm - target_clone
    output_matrix = output_norm - output_clone

    return torch.mean(torch.abs(output_matrix-target_matrix))
    

def contrastive_loss(output,target,comp):
    MAE = torch.mean(torch.abs(output-target))
    ordering = 0
    last_index = 0
    stored_comp = comp[0]
    for i in range(len(comp)):
        curr_comp = comp[i]
        if curr_comp != stored_comp:
            ordering += pairwise_probs(output[last_index:i],target[last_index:i])
            last_index = i
            stored_comp = comp[i]
    
    ordering += pairwise_probs(output[last_index:],target[last_index:])
    
    return MAE + ordering

