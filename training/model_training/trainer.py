from models.PerovskiteOrderingGCNNs_painn.nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
import torch

def trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num):
    if model_type == "Painn":
        best_model = train_painn(model,train_loader,val_loader,hyperparameters,OUTDIR,gpu_num)
    else:
        if contrastive in "model_type":
            loss_fn = get_contrastive_loss()      
        else:
            loss_fn = torch.nn.L1Loss()

        best_model = 

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

def train_CGCNN_e3nn():




def get_contrastive_loss():


