from sacred.experiment import Experiment
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader
from data import IncrementalDataset
from tqdm import tqdm
from model import IncResNet
from PIL import ImageFile
from utils import set_random_seed
ImageFile.LOAD_TRUNCATED_IMAGES = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
writer = SummaryWriter(comment = "malware_classification%resnet34")          
ex = Experiment("malware_classification_resnet34")

@ex.config
def cfg():
    num_epoch_pretrain = 20
    num_epoch_inc_train = 10
    start_num_class = 30
    num_class_per_session = 5
    val_ratio = 0.2
    test_ratio = 0.2
    dataset_name = "bodmas_top50"
    data_folder = "data/bodmas_top50"    
    n_cls = 50
    batch_size=128
    ckpt_save_folder = "./ckpt"  
    lr = 1e-3
    device_ids = [0, 1]
    num_workers=4      
    seed = 11

@ex.command
def train(
    dataset_name, 
    data_folder,
    start_num_class,
    num_class_per_session,
    val_ratio,
    test_ratio,
    n_cls, 
    num_epoch_pretrain,
    num_epoch_inc_train,    
    batch_size,
    device_ids,       
    lr,
    seed,
    ckpt_path = None,    
):      
    set_random_seed(seed)
    inc_data = IncrementalDataset(dataset_name, data_folder, start_num_class=start_num_class, num_class_per_session=num_class_per_session, val_ratio=val_ratio, test_ratio=test_ratio, batch_size=batch_size)
    model = IncResNet(num_block=[3, 4, 6, 3], base_num_classes=30).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    

    start_session = 0
    start_epoch = 0    
    
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)    
        start_session = ckpt['session'] + 1
        start_epoch  = ckpt['epoch'] + 1      
        model.load_state_dict(ckpt['model_state_dict'])   
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])    
        for i in range(start_session):
            if i == 0:
                inc_data.start_task()
            else:
                inc_data.new_task()
        

    for i in range(start_session, 1 + (n_cls - start_num_class) // num_class_per_session):
        if i == 0: # pretraining
            train_set, val_set = inc_data.start_task()    
            _train(model, optimizer, train_set, val_set, num_epoch=num_epoch_pretrain, session_idx=0, start_epoch=start_epoch)                
            start_epoch = 0
                           
        else:                                            
            train_set, val_set = inc_data.new_task()
            # freeze pretrained model
            for name, module in model.named_children():
                if name != "fc" or name != "fc_middle":
                    for param in module.parameters():
                        param.requires_grad = False              
            model.add_classes(num_class_per_session)    
            model = model.to(device) # to device after add new modules
            _train(model, optimizer, train_set, val_set, num_epoch=num_epoch_inc_train, session_idx=i, start_epoch=start_epoch)            
            start_epoch = 0

    writer.close()

    
@ex.capture
def _train(
    model, 
    optimizer,
    train_set, 
    val_set,
    num_epoch,  
    session_idx,
    start_epoch,
    batch_size,     
    num_workers,   
    ckpt_save_folder,    
):

    optim = optimizer
    
    step = 0
    for ep in range(start_epoch, num_epoch):
        # training 
        model.train()
        criterion = nn.CrossEntropyLoss()        
                
        # data
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for idx, data in loop:
            x, y = data
            y = torch.from_numpy(np.asarray(y, dtype=np.long))
            x = x.to(device)                    
            y = y.to(device)

            optim.zero_grad()    
            y_pred = model(x)
            loss = criterion(y_pred, y)              
            loss.backward()
            optim.step()            

            writer.add_scalar(f"Training_Loss_{session_idx}", loss.item(), step)
            step += 1
            loop.set_description(f"SESSION_{session_idx}-Epoch [{ep} / {num_epoch}]")
            loop.set_postfix(loss=loss.item())

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                x, y = data
                y = torch.from_numpy(np.asarray(y, dtype=np.long))
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)
                prediction = torch.argmax(y_pred, 1)
                for i in range(len(y)):
                    correct += (prediction == y).sum()
                    total += len(y)        
            acc = correct / total        
            writer.add_scalar(f"Validation_Acc_{session_idx}", acc, ep)        
            loop.set_postfix(acc=acc)

        # save ckpt
        if not os.path.exists(ckpt_save_folder):
            os.mkdir(ckpt_save_folder)
        torch.save({
            'session': session_idx,
            'epoch': ep,            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()            
        }, os.path.join(ckpt_save_folder, f"ckpt-session_{session_idx}-ep_{ep}-acc_{acc}"))
        
        
if __name__ == "__main__":
    ex.run_commandline()