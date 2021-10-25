from sacred.experiment import Experiment
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from data import IncrementalDataset
from tqdm import tqdm
from model import IncResNet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(comment = "malware_classification%resnet34")       
ex = Experiment("malware_classification_resnet34")

@ex.config
def cfg():
    num_epoch_pretrain = 10
    num_epoch_inc_train = 5
    start_num_class = 20
    num_class_per_session = 5
    val_ratio = 0.2
    test_ratio = 0.2
    dataset_name = "bodmas_top50"
    data_folder = "data/bodmas_top50"
    n_cls = 50

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
    num_epoch_inc_train
):    
    
    inc_data = IncrementalDataset(dataset_name, data_folder, start_num_class=start_num_class, num_class_per_session=num_class_per_session, val_ratio=val_ratio, test_ratio=test_ratio)
    model = IncResNet(num_block=[3, 4, 6, 3], base_num_classes=20).to(device)    
    
    # pretraining
    train_loader, val_loader = inc_data.start_task()    
    _train(model, train_loader, val_loader, num_epoch=num_epoch_pretrain, session_idx=0)    

    # freeze pretrained model
    for name, module in model.named_children():
        if name != "fc":
            for param in module.parameters():
                param.requires_grad = False

    # inctraining         
    for i in range(1, 1 + (n_cls - start_num_class) // num_class_per_session):
        model.add_classes(num_class_per_session)    
        model = model.to(device) # to device after add new modules
        train_loader, val_loader = inc_data.new_task()
        _train(model, train_loader, val_loader, num_epoch=num_epoch_inc_train, session_idx=i)
                
    
    
@ex.capture
def _train(
    model, 
    train_loader, 
    val_loader,
    num_epoch,  
    session_idx  
):
    step = 0
    for ep in range(num_epoch):
        # training 
        model.train()
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
                
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
            loop.set_description(f"Epoch [{ep} / {num_epoch}]")
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
    writer.close()
        
        
if __name__ == "__main__":
    ex.run_commandline()