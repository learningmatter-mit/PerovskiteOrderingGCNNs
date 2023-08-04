import os
from models.PerovskiteOrderingGCNNs_painn.nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from training.loss import contrastive_loss
from training.evaluate import evaluate_model
import torch
from torch.autograd import Variable
from tqdm import tqdm
import time

def trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num,train_eval_loader=None):
    
    hyperparameters["MaxEpochs"] = 100
    
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    if "contrastive" in model_type:
        loss_fn = contrastive_loss 
    else:
        loss_fn = torch.nn.L1Loss()
    
    if model_type == "Painn":
        best_model = train_painn(model,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num)
    else:
        best_model = train_CGCNN_e3nn(model,normalizer,model_type,loss_fn,contrastive_loss,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num,train_eval_loader)

    return best_model, loss_fn


def train_painn(model,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num):

    prop_names = model.output_keys
    loss_fn = loss.build_mae_loss(loss_coef = {prop: 1.0 for prop in prop_names})
    train_metrics = [metrics.MeanAbsoluteError(prop) for prop in prop_names]

    #loss_fn = loss.build_mse_loss(loss_coef = {prop: 1.0 for prop in prop_names})
    #train_metrics = [metrics.MeanSquaredError(prop) for prop in prop_names]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=10**hyperparameters["log_lr"])
    num_epochs = hyperparameters["MaxEpochs"]
    
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

def train_CGCNN_e3nn(model,normalizer,model_type,loss_fn,contrastive_loss_fn,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num,train_eval_loader = None):
### Adapted from https://github.com/ninarina12/phononDoS_tutorial/blob/main/utils/utils_model.py
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    best_validation_error = 99999999
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=10**hyperparameters["log_lr"])
    max_epochs = hyperparameters['MaxEpochs']
    scheduler = ReduceLROnPlateau(
            optimizer,
            patience=hyperparameters["reduceLR_patience"],
            factor=0.5, 
            min_lr=1e-7, 
        )

    results = {}
    history = []
    
    for epoch in range(max_epochs):
        model.train()
        start_time = time.time()
        
        for j, d in tqdm(enumerate(train_loader), total=len(train_loader)):

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


            if model_type == "CGCNN":
                loss = loss_fn(normalizer.denorm(output), target)
            elif model_type == "e3nn_contrastive":
                loss, direct_loss, contrastive_loss = loss_fn(normalizer.denorm(output), d.target, d.comp)
            else:
                loss = loss_fn(normalizer.denorm(output), d.target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time    
    
        model.eval()

        if "e3nn" in model_type and train_eval_loader != None:
            predictions, targets, train_avg_loss = evaluate_model(model, normalizer, model_type, train_eval_loader, contrastive_loss_fn, gpu_num,is_contrastive=True)
            predictions, targets, valid_avg_loss = evaluate_model(model, normalizer, model_type, val_loader, contrastive_loss_fn, gpu_num,is_contrastive=True)

            results = record_keep(history,results,epoch,wall,optimizer,valid_avg_loss,train_avg_loss,model,"contrastive")
            
            if "contrastive" in model_type:
                validation_loss = valid_avg_loss[0]
            else:
                validation_loss = valid_avg_loss[1]

        else:
            predictions, targets, train_avg_loss = evaluate_model(model, normalizer, model_type, train_loader, loss_fn, gpu_num)
            predictions, targets, valid_avg_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num)
        
            results = record_keep(history,results,epoch,wall,optimizer,valid_avg_loss,train_avg_loss,model,"standard")
            validation_loss = valid_avg_loss[0]

        if validation_loss < best_validation_error:
            best_validation_error = validation_loss
            with open(OUTDIR + '/best_model.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step(validation_loss)

    with open(OUTDIR + '/final_model.torch', 'wb') as f:
        torch.save(results, f)

    model_state = torch.load(OUTDIR + '/best_model.torch', map_location=torch.device('cpu'))['state']
    model.load_state_dict(model_state)
    model.to(device)
    return model




def record_keep(history,results,epoch,wall,optimizer,valid_avg_loss,train_avg_loss,model,eval_type):
    if "contrastive" in eval_type:

        history.append({
            'step': epoch,
            'wall': wall,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'valid': {
                'loss': valid_avg_loss[0],
                'direct': valid_avg_loss[1],
                'contrastive': valid_avg_loss[2],
            },
                'train': {
                'loss': train_avg_loss[0],
                'direct': train_avg_loss[1],
                'contrastive': train_avg_loss[2],
            },
        })

        results = {
            'history': history,
            'state': model.state_dict()
        }

        print(f"{epoch+1:4d} ," +
            f"lr = {optimizer.param_groups[0]['lr']:8.8f}  " + 
            f"train loss = {train_avg_loss[0]:8.8f}  " +
            f"train direct = {train_avg_loss[1]:8.8f}  " +
            f"train contrastive = {train_avg_loss[2]:8.8f}  " +
            f"val loss = {valid_avg_loss[0]:8.8f}  " + 
            f"val direct = {valid_avg_loss[1]:8.8f}  " + 
            f"val contrastive = {valid_avg_loss[2]:8.8f}  " + 
            f"time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")
    else:
        history.append({
            'step': epoch,
            'wall': wall,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'valid': {
                'loss': valid_avg_loss[0],
            },
            'train': {
                'loss': train_avg_loss[0],
            },
        })

        results = {
            'history': history,
            'state': model.state_dict()
        }

        print(f"{epoch+1:4d} ," +
            f"lr = {optimizer.param_groups[0]['lr']:8.8f}  " + 
            f"train loss = {train_avg_loss[0]:8.8f}  " +
            f"val loss = {valid_avg_loss[0]:8.8f}  " + 
            f"time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")


    return results
