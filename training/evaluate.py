def evaluate_model(model, normalizer, dataloader, loss_fn, loss_fn_mae, device):
    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mae = 0.
    
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            loss = loss_fn(normalizer.denorm(output), d.target, d.comp).cpu()
            loss_mae = loss_fn_mae(normalizer.denorm(output), d.target).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    
    return loss_cumulative/len(dataloader), loss_cumulative_mae/len(dataloader)