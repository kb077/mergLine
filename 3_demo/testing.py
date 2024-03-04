import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Mesh_faces_dataset_3_test import *
from pointMLP import *


def test ():
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1910"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    previous_check_point_path = '.\\models_pointmlp_biou_alpha'
    previous_check_point_name = 'pointmlp_test_1_best.tar'
    val_list=".\\output\\dataset_test.h5"
    checkpoint_name = 'latest_checkpoint.tar'

    seed = 1
    num_classes = 2
    num_channels=24
    num_workers = 0
    val_batch_size = 1
    
    torch.manual_seed(seed)

    val_dataset = Mesh_Dataset(data_list_path=val_list,
                            num_classes=num_classes,
                            patch_size=13000)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=True,
                            num_workers=num_workers)

    torch.cuda.is_available()

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = pointMLP_seg(num_classes=num_classes, num_channels=num_channels)
    model = nn.DataParallel(model)
    model = model.to(device, dtype=torch.float)
    
    best_val_dsc = 0.0
    checkpoint = torch.load(os.path.join(previous_check_point_path, previous_check_point_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    val_mdsc = checkpoint['val_mdsc']
    del checkpoint
    best_val_dsc = max(val_mdsc)
    print('previous best value: ', best_val_dsc)
        
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    len_x=0
    n_patient = 0
    for k_batch in range (1):
        for j_batch, j_batched_val_sample in enumerate(val_loader):
            if j_batch == k_batch:
                len_x = j_batched_val_sample['len_x'].to(device, dtype=torch.long)[0]
                data_name = j_batched_val_sample['data_name']
                
        real_labels=[-1]*len_x
        print(len(real_labels))
        if not os.path.exists(data_name[0]):
            continue
        else:
            print(data_name)
            
        for i in range(8):
            # validation
            model.eval()
            with torch.no_grad():
                for i_batch, batched_val_sample in enumerate(val_loader):
                    if i_batch == k_batch:
                        inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                        selected_idx = batched_val_sample['selected_idx']

                        val_dataset.update(selected_idx,i)
                        selected_idx = selected_idx.to(device, dtype=torch.long)
                        
                        outputs = model(inputs)

                        probabilities = F.softmax(outputs, dim=1)
                        predicted_labels = torch.argmax(probabilities, dim=2)
                        #print(selected_idx)
                        for k , si in enumerate(selected_idx[0]):
                            real_labels[si]=predicted_labels[0][k]                  

        real_labels = torch.tensor(real_labels).cpu().numpy()
        n_patient+=1
        np.save(data_name[0]+"-labels-pointmlp-full-dataset-biou-alpha.npy", real_labels)
