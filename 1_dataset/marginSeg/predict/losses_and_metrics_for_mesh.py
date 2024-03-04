import torch
import numpy as np
from pointMLP import *
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def B_IoU(y_pred,y_true,inputs):
    k=10
    n_classes = y_pred.shape[-1]
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)
    
    xyz = inputs[:,9:12,:].permute(0, 2, 1)
    idx = knn_point(k,xyz,xyz).reshape(-1,k)
    mbiou=0.0
    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        neighbor_pred_labels = pred_flat[idx]
        neighbor_true_labels = true_flat[idx]
        bp =  pred_flat*k != torch.sum(neighbor_pred_labels,axis=1)  
        bl =  true_flat*k != torch.sum(neighbor_true_labels,axis=1)
        bl = bl.int()
        bp = bp.int()
        intersection  = (bl * bp).sum()   
        blbp = bl + bp
        union = blbp.to(bool).int().sum()
        mbiou+= intersection / union
    return mbiou/n_classes

def B_centroids_distance(y_pred,y_true,inputs):
    k=10
    n_classes = y_pred.shape[-1]
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)
    
    xyz = inputs[:,9:12,:].permute(0, 2, 1)
    xyz_reshape = xyz.reshape(-1,xyz.shape[-1])
    idx = knn_point(k,xyz,xyz).reshape(-1,k)
    mbcd=0.0
    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        neighbor_pred_labels = pred_flat[idx]
        neighbor_true_labels = true_flat[idx]
        bp =  pred_flat*k != torch.sum(neighbor_pred_labels,axis=1)  
        bl =  true_flat*k != torch.sum(neighbor_true_labels,axis=1)
        bl = bl.int()
        bp = bp.int()
        bl_xyz = (xyz_reshape*bl.unsqueeze(1)).sum(axis=0)/(bl.sum()+1e-10)
        bp_xyz = (xyz_reshape*bp.unsqueeze(1)).sum(axis=0)/(bp.sum()+1e-10)
        mbcd+= F.smooth_l1_loss(bl_xyz,bp_xyz) #torch.sqrt(torch.sum((bl_xyz - bp_xyz)**2))# 
    return mbcd/n_classes

def weighting_BIoU(y_pred,y_true,inputs):
    mbiou =0.0
    k=32
    _eps = 1e-12
    temperature =1
    weight = 0.1
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)
    
    xyz = inputs[:,9:12,:].permute(0, 2, 1)
    #fps_idx = farthest_point_sample(xyz, npoint=13000).long()
    #new_xyz = index_points(xyz, fps_idx)
    #idx = knn_point(32, xyz, new_xyz)
    idx = knn_point(k,xyz,xyz)
    #grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
    features = inputs.reshape(inputs.shape[1],-1)
    idx = idx.permute(2, 0, 1)
    idx=idx.reshape(idx.shape[0],-1)
    neighbor_feature = features[ :, idx]

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        posmask = pred_flat == true_flat
        nsample = len(pred_flat)
        point_mask = torch.sum(posmask.int(), -1)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)
        
        posmask = posmask[point_mask]
        
        selected_indices = torch.nonzero(posmask[0]).squeeze()
        # features = features[:,selected_indices]
        # neighbor_feature = neighbor_feature[:,:,selected_indices]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]
        
        #l2 dist
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-2) + _eps) # [m, nsample]
        #loss
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if temperature is not None:
            dist = dist / temperature
        exp = torch.exp(dist)
        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        
        w = weight
        loss = torch.mean(loss)
        loss *= float(w)
        mbiou+=loss
    return mbiou

def weighting_DSC(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mdsc = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc += w*((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
        
    return mdsc


def weighting_SEN(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    msen = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        msen += w*((intersection + smooth) / (true_flat.sum() + smooth))
        
    return msen


def weighting_PPV(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mppv = 0.0
    n_classes = y_pred.shape[-1] 

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mppv += w*((intersection + smooth) / (pred_flat.sum() + smooth))
        
    return mppv

   
def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[-1]
    
    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
       
        # with weight
        w = class_weights[c]/class_weights.sum()
        loss += w*(1 - ((2. * intersection + smooth) /
                         (pred_flat.sum() + true_flat.sum() + smooth)))
       
    return loss

def DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc


def batch_DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    dsc = []

    # added the one-hot-labeling of the 
    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = one_hot[:, :, c].reshape(-1)
            true_flat = y_true[:, :, c].reshape(-1)
            #pred_flat = y_pred[:, c].reshape(-1)
            #true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = one_hot[:, :, c].reshape(-1)
            true_flat = y_true[:, :, c].reshape(-1)
            #pred_flat = y_pred[:, c].reshape(-1)
            #true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc


def SEN(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
        
    return sen


def PPV(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
        
    return 

# from unet_isbi data
#from post_processing import *
import numpy as np
from PIL import Image
import glob as gl
import numpy as np
from PIL import Image
import torch

def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)
    return accuracy/len(np_ims[0].flatten())


def accuracy_check2(y_pred, y_true):
    print('y_pred size: ', y_pred.size())
    #n_pts = y_pred.shape[1] * y_pred.shape[0] # multiply the number of points per mesh with batch size
    n_pts = y_pred.shape[1] 
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    acc = 0.0
    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        #print('intersection: ', intersection)
        acc += intersection

    return acc/n_pts


def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc/batch_size


"""
def accuracy_compare(prediction_folder, true_mask_folder):
    ''' Output average accuracy of all prediction results and their corresponding true masks.
    Args
        prediction_folder : folder of the prediction results
        true_mask_folder : folder of the corresponding true masks
    Returns
        a tuple of (original_accuracy, posprocess_accuracy)
    '''

    # Bring in the images
    all_prediction = gl.glob(prediction_folder)
    all_mask = gl.glob(true_mask_folder)

    # Initiation
    num_files = len(all_prediction)
    count = 0
    postprocess_acc = 0
    original_acc = 0

    while count != num_files:

        # Prepare the arrays to be further processed.
        prediction_processed = postprocess(all_prediction[count])
        prediction_image = Image.open(all_prediction[count])
        mask = Image.open(all_mask[count])

        # converting the PIL variables into numpy array
        prediction_np = np.asarray(prediction_image)
        mask_np = np.asarray(mask)

        # Calculate the accuracy of original and postprocessed image
        postprocess_acc += accuracy_check(mask_np, prediction_processed)
        original_acc += accuracy_check(mask_np, prediction_np)
        # check individual accuracy
        print(str(count) + 'th post acc:', accuracy_check(mask_np, prediction_processed))
        print(str(count) + 'th original acc:', accuracy_check(mask_np, prediction_np))

        # Move onto the next prediction/mask image
        count += 1

    # Average of all the accuracies
    postprocess_acc = postprocess_acc / num_files
    original_acc = original_acc / num_files

    return (original_acc, postprocess_acc)
"""

# Experimenting
if __name__ == '__main__':
    '''
    predictions = 'result/*.png'
    masks = '../data/val/masks/*.png'

    result = accuracy_compare(predictions, masks)
    print('Original Result :', result[0])
    print('Postprocess result :', result[1])
    '''
