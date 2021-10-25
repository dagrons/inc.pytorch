import torch
import random
from torch.utils.data import * 
from PIL import Image
import os 
import numpy as np

class MalwareImgDataset(Dataset):
    """
    Dataset for Malware Imgs
    Dataset 抽象了数据的实际读取方式

    Attributes
    ----------  
    n_cls: number of unique labels
    images: the file path of images
    labels: the label of images

    Methods: 
    --------
    __len__():
        The number of images     
    __getitem__(idx): 
        Return the tensor data indexed by idx
    """

    def __init__(self, data_folder):
        self.data = []
        self.labels = []     
        self.setup_data(data_folder)            
        self.n_cls = 50      
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    
    def setup_data(self, data_folder):
        for label in os.listdir(data_folder):
            label_path = os.path.join(data_folder, label)    
            for fname in os.listdir(label_path):
                fpath = os.path.join(label_path, fname)
                self.data.append(fpath)
                self.labels.append(label)                       
        idx2class = list(sorted(set(self.labels)))
        self.labels = [idx2class.index(y) for y in self.labels]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]        
        label = self.labels[idx]

        img = Image.open(img_path)
        img = img.resize((64, 64))
        img_mat = np.asarray(img, dtype=np.float32)
        img_mat = np.reshape(img_mat, (1, 64, 64)) # can be replaced with unsqueeze
        img_mat = np.repeat(img_mat, 3, axis=0) # (3, 224, 224)
        img_tensor = torch.from_numpy(img_mat)

        return img_tensor, label

class DummyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]        
        label = self.labels[idx]

        img = Image.open(img_path)
        img = img.resize((64, 64))
        img_mat = np.asarray(img, dtype=np.float32)
        img_mat = np.reshape(img_mat, (1, 64, 64)) # can be replaced with unsqueeze
        img_mat = np.repeat(img_mat, 3, axis=0) # (3, 224, 224)
        img_tensor = torch.from_numpy(img_mat)

        return img_tensor, label
        
def get_dataset(dataset_name, data_folder):
    """
    A factory method to return the corresponding dataset identified by dataset_name

    Parameters
    ----------
    dataset_name: dataset name, eg: BOMDAS_IMG_TOP50            
    data_folder: where the data is located

    Returns
    ---------
    dataset: 
        The corresponding dataset        
    """
    if dataset_name == "bodmas_top50":
        return MalwareImgDataset(data_folder)    
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

class NoMoreSessionsException(Exception):
    pass

class IncrementalDataset:
    """
    Incremental Dataset as a wrapper for base dataset

    注意：validation是用来epoch-准确率曲线从而帮助我们调参的，test是用来衡量最终准确率的

    对于每一个session:
        training set 只有当前session的training data 和 memory data
        validation set 应该包含之前见过的所有类别

    test set 用来衡量最终的正确率

    Attributes
    -----------
    start_num_class: num_class for pretraining 
    num_class_per_session: num_class per session    
    data_train: data for training 
    data_val: data for validation
    data_test: data for testing 
    targets_train: targets for training data
    targets_val: targets for validation data
    targets_test: targets for testing data
    n_cls: number of total classes
    class_order: class order
    current_class_idx: current class idx in the class order 
    data_memory: a small percentage of data from pervious session for training     
    targets_memory: memory data targets

    Methods
    -------
    new_task():
        generate train_dataset and test_dataset for incremental sessions
    start_task():
        generate train_dataset and test_dataset for pretraining session
    """
    def __init__(self, dataset_name, data_folder, start_num_class, num_class_per_session, val_ratio=0.2, test_ratio=0.2, batch_size=64):        
        # data folder and dataset name
        self.data_foler = data_folder
        self.dataset_name = dataset_name         
        self.start_num_class = start_num_class # num_classes for start session
        self.num_class_per_session = num_class_per_session # num_classes for incremental sessions     \    

        # split dataset into train, val and test
        dataset = get_dataset(dataset_name, data_folder)
        self.data_train, self.targets_train, self.data_val, self.targets_val, self.data_test, self.targets_test = self.split_train_val_test(dataset, val_ratio, test_ratio)
        
        # get class order        
        self.n_cls = dataset.n_cls                
        self.class_order = self.get_class_order(dataset.n_cls)

        # current class idx
        self.current_class_idx = 0

        # memory data
        self.data_memory = None
        self.targets_memory = None

        self.num_sessions = 1 + (self.n_cls - self.start_num_class) // num_class_per_session
        self.current_session = 0

        # batch size
        self.batch_size = batch_size

    def start_task(self):
        """
        Sample start task for pretraining 
        """
        selected_train_classes = self.class_order[self.current_class_idx:self.current_class_idx + self.start_num_class]     
        selected_val_classes = self.class_order[self.current_class_idx:self.current_class_idx + self.start_num_class]             
        x_train, y_train = self.select_from(self.data_train, self.targets_train, selected_train_classes)
        y_train = [self.class_order.index(i) for i in y_train]
        # D^hat_t = D_t + M                
        if self.data_memory is not None:
            x_memory_train, y_memory_train = self.select_from(self.data_memory, self.targets_memory, selected_val_classes)
            x_train = np.concatenate([x_train, x_memory_train])
            y_train = np.concatenate([y_train, y_memory_train])        
        x_val, y_val = self.select_from(self.data_val, self.targets_val, selected_val_classes)  
        y_val = [self.class_order.index(i) for i in y_val]                  
        self.current_class_idx += self.start_num_class    
        self.current_session += 1                            
        train_set = DummyDataset(x_train, y_train)        
        val_set = DummyDataset(x_val, y_val)        
        return train_set, val_set

    def new_task(self):
        """
        Sample a new task, including train_loader, val_loader and test_loader        
        """        
        if self.current_session >= self.num_sessions:
            raise NoMoreSessionsException("No more sessions!")

        selected_train_classes = self.class_order[self.current_class_idx:self.current_class_idx + self.num_class_per_session]        
        selected_val_classes = self.class_order[:self.current_class_idx + self.num_class_per_session]        
        x_train, y_train = self.select_from(self.data_train, self.targets_train, selected_train_classes)  
        y_train = [self.class_order.index(i) for i in y_train]
        # D^hat_t  = D_t + M 
        if self.data_memory is not None:
            x_memory_train, y_memory_train = self.select_from(self.data_memory, self.targets_memory, selected_val_classes)
            x_train = np.concatenate([x_train, x_memory_train])
            y_train = np.concatenate([y_train, y_memory_train])        
        x_val, y_val = self.select_from(self.data_val, self.targets_val, selected_val_classes)        
        y_val = [self.class_order.index(i) for i in y_val]
        self.current_class_idx += self.num_class_per_session        
        self.current_session += 1     
        train_set = DummyDataset(x_train, y_train)        
        val_set = DummyDataset(x_val, y_val)        
        return train_set, val_set

    
    @staticmethod
    def select_from(x, y, selected_classes):
        """
        Select data by selected_classes 
        """
        data = []
        targets = []        
        for cls in selected_classes:
            idxes = np.where(y == cls)[0]            
            data.append(x[idxes])
            targets.append(y[idxes])
        
        data = np.concatenate(data)
        targets = np.concatenate(targets)
        return data, targets

    @staticmethod
    def get_class_order(n_cls):
        """
        Seed a class order for incremental sessions 
        """
        order = [i for i in range(n_cls)]        
        random.shuffle(order)   
        return order             
        
    @staticmethod
    def split_train_val_test(dataset, val_ratio, test_ratio):        
        """
        Split dataset into train, val, test

        Returns
        -------
        x_train, y_train: train set
        x_val, y_val: validation set
        x_test, y_test: test set
        """        
        x, y = dataset.data, dataset.labels                        
        # shuffle x and y first 
        shuffled_indexes = np.random.permutation(len(x))
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        # split x and y into train_set and validation_set                
        x_train, y_train = [], []    
        x_val, y_val = [], []
        x_test, y_test = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * val_ratio) # number of validation data
            nb_test_elts = int(class_indexes.shape[0] * test_ratio) # number of test data

            val_indexes = class_indexes[:nb_val_elts] 
            test_indexes = class_indexes[nb_val_elts:nb_val_elts + nb_test_elts]
            train_indexes = class_indexes[nb_val_elts + nb_test_elts:]            

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_test.append(x[test_indexes])
            y_test.append(y[test_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        # !list, convert list to ndarray
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)   
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":    
    from utils import set_random_seed
    set_random_seed()
    inc_dataset = IncrementalDataset("bodmas_top50", "data/bodmas_top50", 20, 5, 0.2, 0.2)
    # start task
    train_loader, val_loader = inc_dataset.start_task()
    print (f"train: {len(train_loader)}, val: {len(val_loader)}")
    # new task
    for i in range(1, inc_dataset.num_sessions):
        train_loader, val_loader = inc_dataset.new_task()
        print (f"train: {len(train_loader)}, val: {len(val_loader)}")