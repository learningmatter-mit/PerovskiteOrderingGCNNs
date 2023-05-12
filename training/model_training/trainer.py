from models.PerovskiteOrderingGCNNs_painn.nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training.loss import contrastive_loss
from training.evaluate import evaluate_model
import torch
from torch.autograd import Variable
import time

def trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num):
    if model_type == "Painn":
        best_model = train_painn(model,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num)
    else:
        if contrastive in "model_type":
            loss_fn = contrastive_loss 
        else:
            loss_fn = torch.nn.L1Loss()

        best_model = train_CGCNN_e3nn(model,normalizer,model_type,loss_fn,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num)

    return best_model


def train_painn(model,model,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num):

    prop_names = model.output_keys
    loss_fn = loss.build_mae_loss(loss_coef = {prop: 1.0 for prop in prop_names})
    train_metrics = [metrics.MeanAbsoluteError(prop) for prop in prop_names]

    #loss_fn = loss.build_mse_loss(loss_coef = {prop: 1.0 for prop in prop_names})
    #train_metrics = [metrics.MeanSquaredError(prop) for prop in prop_names]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=10**hyperparameters["log10_lr"])
    num_epochs = hyperparameters["epochs"]
    
    train_hooks = [
        hooks.MaxEpochHook(num_epochs),
        hooks.CSVHook(
            OUTDIR, 
            metrics=train_metrics
        ), 
        hooks.PrintingHook(
            OUTDIR, 
            metrics=train_metrics, 
            separator = ' | ', 
            time_strf='%M:%S'
        ), 
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer, 
            patience=hyperparameters["reduceLR_patience"], 
            factor=0.5, 
            min_lr=1e-7, 
            window_length=1, 
            stop_after_min=True)
    ]
    
    T = Trainer(
        model_path=OUTDIR, 
        model=model, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks,
        mini_batches=1
    )
    
    T.train(device=gpu_num, n_epochs=num_epochs)

    return T.get_best_model()

def train_CGCNN_e3nn(model,normalizer,model_type,loss_fn,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num):
### Adapted from https://github.com/ninarina12/phononDoS_tutorial/blob/main/utils/utils_model.py
    device_name = "cuda:" + gpu_num
    device = torch.device(device_name)
    
    best_validation_error = 99999999
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])
    max_epochs = hyperparameters['MaxEpochs']
    scheduler = ReduceLROnPlateau(
            optimizer,
            patience=hyperparameters["reduceLR_patience"],
            factor=0.5, 
            min_lr=1e-7, 
            window_length=1, 
            stop_after_min=True
        )

    results = {}
    history = []
    
    for epoch in range(max_epochs):
        model.train()
        
        for j, d in tqdm(enumerate(train_loader), total=len(train_loader)):

            if model_type == "CGCNN":
                input_struct = d[0]
                target = d[1]
                input_var = (Variable(input_struct[0].cuda(non_blocking=True)),
                             Variable(input_struct[1].cuda(non_blocking=True)),
                             input_struct[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input_struct[3]])
                output = model(*input_var)
            else:
                d.to(device)
                output = model(d)

            if model_type == "CGCNN":
                loss = loss_fn(normalizer.denorm(output), target)
            elif model_type == "e3nn_contrastive":
                loss = loss_fn(normalizer.denorm(output), d.target, d.comp)
            else:
                loss = loss_fn(normalizer.denorm(output), d.target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time    
    
        model.eval()
        results, targets, valid_avg_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num)
        results, targets, train_avg_loss = evaluate_model(model, normalizer, model_type, train_loader, loss_fn, gpu_num)

        history.append({
            'step': step,
            'wall': wall,
            'valid': {
                'loss': valid_avg_loss,
            },
            'train': {
                'loss': train_avg_loss,
            },
        })

        results = {
            'history': history,
            'state': model.state_dict()
        }

        print(f"{step+1:4d} ," +
              f"lr = {optimizer.param_groups[0]['lr']:8.8f}  " + 
              f"train loss = {train_avg_loss:8.8f}  " +
              f"val loss = {valid_avg_loss:8.8f}  " + 
              f"time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

        if valid_avg_loss < best_validation_error:
            best_validation_error = valid_avg_loss
            with open(OUTDIR + '/best_model.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step(valid_avg_loss)

    with open(OUTDIR + '/final_model.torch', 'wb') as f:
        torch.save(results, f)

    model_state = torch.load(OUTDIR + '/best_model.torch', map_location=device)['state']
    model.load_state_dict(model_state)
    return model





