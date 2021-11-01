import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import json
import os
import time
from PIL import Image
import random
import torchvision
from torchvision import transforms
#from data_utils.uv_map_generate_freihand import *
from uv_map_generate_freihand import *
import copy
import torch
import time
import math
import matplotlib.pyplot as plt
import matplotlib
import chumpy as ch
from chumpy.ch import MatVecMult
from utils.mano_core.mano_loader import load_model
#from utils.mano_utils import get_keypoints_from_mesh_ch


#########################################################################################
def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd
def cv2pil(cv_img):
    return Image.fromarray(cv2.cvtColor(np.uint8(cv_img), cv2.COLOR_BGR2RGB))
""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'
def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)

def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return cv2.imread(img_rgb_path)


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    # mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)
    scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
    verts_path = os.path.join(base_path, '%s_verts.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    # mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)
    scale_list = json_load(scale_path)
    verts_list = json_load(verts_path)

    # should have all the same length
    # assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'
    assert len(K_list) == len(scale_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    # return list(zip(K_list, mano_list, xyz_list, scale_list))
    return list(zip(K_list, xyz_list, scale_list, verts_list))

def process_bbox(bbox, img_width, img_height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = 256 / 256
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox

def rotate(origin, point, angle, scale):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + scale * math.cos(angle) * (px - ox) - scale * math.sin(angle) * (py - oy)
    qy = oy + scale * math.sin(angle) * (px - ox) + scale *  math.cos(angle) * (py - oy)

    return qx, qy



def processing_augmentation(image,pose3d,pointcloud):  #scale, transiltion， uv argumentation is different from the xyz vertices
    randScaleImage = np.random.uniform(low=0.8, high=1.0)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0] #change the rotation to
    rotMat = cv2.getRotationMatrix2D((128, 128), -180.0 * randAngle / math.pi,
                                     randScaleImage)  # change image later together with translation

    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
    #rotation
    (pose3d[:, 0], pose3d[:, 1]) = rotate((128,128), (pose3d[:, 0], pose3d[:, 1]), randAngle, randScaleImage)
    (pointcloud[:, 0], pointcloud[:, 1]) = rotate((128,128), (pointcloud[:, 0], pointcloud[:, 1]), randAngle, randScaleImage)
    pose3d[:,0] = pose3d[:,0] + randTransX
    pose3d[:,1] = pose3d[:,1] + randTransY
    pointcloud[:,0] = pointcloud[:,0] + randTransX
    pointcloud[:,1] = pointcloud[:,1] + randTransY

    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY
    image = cv2.warpAffine(image, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    image = np.reshape(image, [256, 256, 3])
    return image, pose3d,pointcloud



# def processing_augmentation(image,pose3d,pointcloud):  #scale, transiltion， uv argumentation is different from the xyz vertices
#     randScaleImage = np.random.uniform(low=0.8, high=1.0)
#     #randScaleImage = np.random.uniform(low=0.8, high=1.0)
#     #pointcloud = pointcloud*randScaleImage
#     pose3d = np.reshape(pose3d, [21, 3])
#     randAngle = 2 * math.pi * np.random.rand(1)[0]/10.0 #change the rotation to
#     rotMat = cv2.getRotationMatrix2D((128, 128), -180.0 * randAngle / math.pi,
#                                      randScaleImage)  # change image later together with translation
# 
#     #(pose3d[:, 0], pose3d[:, 1]) = rotate((pose3d[3,0], pose3d[3,1]), (pose3d[:, 0], pose3d[:, 1]), randAngle)
#     #(pointcloud[:, 0], pointcloud[:, 1]) = rotate((pose3d[3,0], pose3d[3,1]), (pointcloud[:, 0], pointcloud[:, 1]), randAngle)
# 
#     (pose3d[:, 0], pose3d[:, 1]) = rotate((128,128), (pose3d[:, 0], pose3d[:, 1]), randAngle)
#     (pointcloud[:, 0], pointcloud[:, 1]) = rotate((128,128), (pointcloud[:, 0], pointcloud[:, 1]), randAngle)
# 
#     randTransX = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
#     randTransY = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 40.0), -40.0)
#     rotMat[0, 2] += randTransX
#     rotMat[1, 2] += randTransY
#     #rotMat[0, 2] += 0.0
#     #rotMat[1, 2] += 0.0
#     image = cv2.warpAffine(image, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
#     #pose3d = np.reshape(pose3d, [63])
#     image = np.reshape(image, [256, 256, 3])
#     return image, pose3d,pointcloud

def imcrop(img, center, crop_size):
    x1 = int(np.round(center[0]-crop_size))
    y1 = int(np.round(center[1]-crop_size))
    x2 = int(np.round(center[0]+crop_size))
    y2 = int(np.round(center[1]+crop_size))

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    if img.ndim < 3: # for depth
        img_crop = img[y1:y2, x1:x2]
    else: # for rgb
        img_crop = img[y1:y2, x1:x2, :]

    return img_crop

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    if img.ndim < 3: # for depth
        borderValue = [0]
    else: # for rgb
        borderValue = [127, 127, 127]

    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2
def rgb_processing(rgb_img):
    # in the rgb image we add pixel noise in a channel-wise manner
    noise_factor = 0.4
    pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    return rgb_img
generate_uv  = Generate_uv(UV_height=256, UV_width=256)
class FreiHAND(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert self.mode in ['training', 'evaluation'], 'mode error'
        # load annotations
        self.anno_all = load_db_annotation(root, 'training')
        self.transform = torchvision.transforms.Compose([transforms.ToTensor()])
        db = COCO(os.path.join(root, 'data/freihand_train_coco.json')) ##导入的数据中 长宽都是一样的
        self.bbox = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            bbox = process_bbox(np.array(ann['bbox']), 224, 224)
            self.bbox.append(bbox)
        use_mean_pose = True
        self.model = load_model('/home/ziwei/densepose/freihand/freihand-master/data/MANO_RIGHT.pkl', ncomps=45, flat_hand_mean=not use_mean_pose,
                                use_pca=False)

        with open('/home/ziwei/densepose/freihand/freihand-master/data/MANO_RIGHT.pkl', 'rb') as f:
            smpl_data = pickle.load(f,encoding='latin1')
        self.Jreg = smpl_data['J_regressor']
        #print(self.Jreg.shape)



    def get_keypoints_from_mesh_np(self,mesh_vertices, keypoints_regressed):
        """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
        kpId2vertices = {
            4: [744],  # ThumbT
            8: [320],  # IndexT
            12: [443],  # MiddleT
            16: [555],  # RingT
            20: [672]  # PinkT
        }
        keypoints = [0.0 for _ in range(21)]  # init empty list

        # fill keypoints which are regressed
        mapping = {0: 0,  # Wrist
                   1: 5, 2: 6, 3: 7,  # Index
                   4: 9, 5: 10, 6: 11,  # Middle
                   7: 17, 8: 18, 9: 19,  # Pinky
                   10: 13, 11: 14, 12: 15,  # Ring
                   13: 1, 14: 2, 15: 3}  # Thumb

        for manoId, myId in mapping.items():
            keypoints[myId] = keypoints_regressed[manoId, :]

        # get other keypoints from mesh
        for myId, meshId in kpId2vertices.items():
            keypoints[myId] = np.mean(mesh_vertices[meshId, :], 0)

        keypoints = np.vstack(keypoints)

        return keypoints


    def __len__(self):
        #32560 130240
        return len(self.anno_all)*4
        #return 30000*4
        #return 100
        #return 1

        #return 120

    def __getitem__(self, id):
        idx = id % 32560
        img_idx = id // 32560

        #print(idx,img_idx)
        if img_idx == 0:
            version = 'gs'
        elif img_idx == 1:
            version = 'hom'
        elif img_idx == 2:
            version = 'sample'
        else:
            version = 'auto'
        img = read_img(idx, self.root, 'training', version)
        #image_crop, img2bb_trans, bb2img_trans, rot, _, inv_trans_joint = augmentation(img, self.bbox[idx], 'training', exclude_flip=True)  # FreiHAND dataset only contains right hands. do not perform flip aug.
        bbox = self.bbox[idx]
        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        crop_center_rgb = np.array([[bb_c_x,bb_c_y ]])
        crop_size_rgb = bb_width / 2

        image_crop = imcrop(img, crop_center_rgb[0], crop_size_rgb)
        image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        # annotation for this frame
        K, xyz, scale, verts = self.anno_all[idx]
        K, xyz, scale, verts = [np.array(x) for x in [K, xyz, scale, verts]]

        pose_uvd = xyz2uvd(xyz, K)
        pose_uvd_relatived = copy.deepcopy(pose_uvd)
        pose_uvd_relatived[:,2] = pose_uvd_relatived[:,2] - pose_uvd[0,2]
        pose_uvd_relatived_ortho = copy.deepcopy(pose_uvd_relatived)
        pose_uvd_relatived_ortho[:,0] = (pose_uvd_relatived_ortho[:,0] - crop_center_rgb.reshape(1,2)[0,0] + crop_size_rgb)*(256.0/(crop_size_rgb*2)) #convert the x
        pose_uvd_relatived_ortho[:,1] = (pose_uvd_relatived_ortho[:,1] - crop_center_rgb.reshape(1,2)[0,1] + crop_size_rgb)*(256.0/(crop_size_rgb*2)) #convert the y

        #vertices, change to camera distance
        vertices_uvd = xyz2uvd(verts, K)
        vertices_uvd_relatived = copy.deepcopy(vertices_uvd)
        vertices_uvd_relatived[:,2] = vertices_uvd_relatived[:,2] - pose_uvd[0,2]
        #crop to 255
        vertices_uvd_relatived_ortho = copy.deepcopy(vertices_uvd_relatived)
        vertices_uvd_relatived_ortho[:,0] = (vertices_uvd_relatived_ortho[:,0] - crop_center_rgb.reshape(1,2)[0,0]) #for the left one, no need clip it
        vertices_uvd_relatived_ortho[:,0] = (vertices_uvd_relatived_ortho[:,0] + crop_size_rgb)*(256.0/(crop_size_rgb*2)) #convert the x
        vertices_uvd_relatived_ortho[:,1] = (vertices_uvd_relatived_ortho[:,1] - crop_center_rgb.reshape(1,2)[0,1] + crop_size_rgb)*(256.0/(crop_size_rgb*2)) #convert the y


        if np.random.rand()<0.8:
            image_crop, pose_uvd_relatived_ortho,vertices_uvd_relatived_ortho = processing_augmentation(image_crop, pose_uvd_relatived_ortho,vertices_uvd_relatived_ortho)
        vertices_uvd_relatived_ortho[:,:2] = vertices_uvd_relatived_ortho[:,:2]/256.0
        vertices_uvd_relatived_ortho[:,2] = vertices_uvd_relatived_ortho[:,2]*10

        uv_map, verts_backup, face = generate_uv.get_UV_map(vertices_uvd_relatived_ortho, dilate=True)

        uvd_relatived_ortho = copy.deepcopy(vertices_uvd_relatived_ortho)
        uvd_relatived_ortho[:,:2] = uvd_relatived_ortho[:,:2]*256.0
        uvd_relatived_ortho[:,2] = uvd_relatived_ortho[:,2]/10.0
        uvd_relatived_ortho[:, 0] = uvd_relatived_ortho[:, 0] / (256.0 / (crop_size_rgb * 2)) - crop_size_rgb + \
                                    crop_center_rgb.reshape(1, 2)[0, 0]
        uvd_relatived_ortho[:, 1] = uvd_relatived_ortho[:, 1] / (256.0 / (crop_size_rgb * 2)) - crop_size_rgb + \
                                    crop_center_rgb.reshape(1, 2)[0, 1]
        uvd_relatived_ortho[:, 2] = uvd_relatived_ortho[:, 2] + pose_uvd[0, 2]
        vertices_cam = uvd2xyz(uvd_relatived_ortho, K)

        
        pose_relatived_ortho = copy.deepcopy(pose_uvd_relatived_ortho)
        pose_relatived_ortho[:, 0] = pose_relatived_ortho[:, 0] / (256.0 / (crop_size_rgb * 2)) - crop_size_rgb + \
                                    crop_center_rgb.reshape(1, 2)[0, 0]
        pose_relatived_ortho[:, 1] = pose_relatived_ortho[:, 1] / (256.0 / (crop_size_rgb * 2)) - crop_size_rgb + \
                                    crop_center_rgb.reshape(1, 2)[0, 1]
        pose_relatived_ortho[:, 2] = pose_relatived_ortho[:, 2] + pose_uvd[0, 2]
        Pose_cam = uvd2xyz(pose_relatived_ortho, K)
        
        #normalization uv /255, d /scale

        # cloud_resample = generate_uv.resample(uv_map)
        # cloud_resample[:,:2] = cloud_resample[:,:2]*256.0
        # cloud_resample[:,2] = cloud_resample[:,2]/10.0
        # cloud_resample[:,2] = cloud_resample[:,2] + pose_uvd[0, 2]

        #uv = uv_map.max(axis=2)
        #binary_mask = np.where(uv > 0, 1., 0.)
        #binary_mask = (binary_mask * 255).astype(np.uint8)
        # #
        # np.save("uv_template_mask1.npy", binary_mask)
        vertices_uvd_relatived_ortho[:, :2] = vertices_uvd_relatived_ortho[:, :2] * 256.0
        vertices_uvd_relatived_ortho[:, 2] = vertices_uvd_relatived_ortho[:, 2] / 10

        image_crop = cv2.cvtColor(np.uint8(image_crop), cv2.COLOR_BGR2RGB)
        #cjitter=torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        image_crop = rgb_processing(image_crop)
        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
        img = image_trans(torchvision.transforms.ToPILImage()((img).astype(np.uint8)))
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))
        #return image_crop, uv_map, crop_center_rgb, crop_size_rgb, verts, pose_uvd_relatived_ortho, K, bbox, vertices_cam, binary_mask, cloud_resample
        #batch_img_original, batch_image, batch_uv, batch_vertex_crop, batch_pose3d, batch_vertices, batch_K,
         #          batch_crop_size, batch_crop_center
        return img, image_crop, uv_map, vertices_uvd_relatived_ortho, Pose_cam, vertices_cam,K, crop_size_rgb, \
               crop_center_rgb

class FreiHAND_test(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert self.mode in [ 'evaluation'], 'mode error'
        # load annotations
        self.scale_list = json_load(os.path.join(self.root, '%s_scale.json' % self.mode))
        self.K_list = json_load(os.path.join(self.root, '%s_K.json' % self.mode))
        bbox = os.path.join(root, 'data/bbox_root_freihand_output.json')
        self.bbox_root_result = {}
        with open(bbox) as f:
            annot = json.load(f)
        for i in range(len(annot)):
            self.bbox_root_result[i] = {'bbox': np.array(annot[i]['bbox']),
                                        'root': np.array(annot[i]['root_cam'])}
        print("Get bounding box and root from " + bbox)
    def __getitem__(self, idx):
        img = read_img(idx, self.root, self.mode)
        #img = cv2pil(img)
        bbox = self.bbox_root_result[idx]['bbox']
        root =  self.bbox_root_result[idx]['root']
        root = np.array(root).reshape(1,3) # 这里的root是手腕的点
        K = self.K_list[idx]
        K= np.array(K)

        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        crop_center_rgb = np.array([[bb_c_x,bb_c_y ]])
        crop_size_rgb = bb_width / 2

        image_crop = imcrop(img, crop_center_rgb[0], crop_size_rgb)
        image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

        # annotation for this frame
        #uvd_root = xyz2uvd(root, K)
        image_crop = cv2.cvtColor(np.uint8(image_crop), cv2.COLOR_BGR2RGB)

        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
        img = image_trans(torchvision.transforms.ToPILImage()((img).astype(np.uint8)))

        return img, image_crop, crop_center_rgb, crop_size_rgb, K, root

    def __len__(self):
        return 3960



inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5])

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path  = "/mnt/data/ziwei/Freihand/"
    batch_size = 1
    #root, mode
    dataset = FreiHAND(root = path, mode="training")
    trainloader_synthesis = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    dataset = FreiHAND_validation(root = path, mode="training")
    trainloader_valuation = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    dataset_test = FreiHAND_test(root = path, mode="evaluation")
    test_synthesis = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

    for step, ( img, image_crop, xyz, verts, K, crop_size_rgb, crop_center_rgb
                ) in enumerate(trainloader_valuation):
        print(step)
        # batch_image = inv_normalize(image_crop[0])
        # image_crop = batch_image.cpu().detach().numpy().transpose(1, 2, 0)
        #
        # batch_image = inv_normalize(img[0])
        # img = batch_image.cpu().detach().numpy().transpose(1, 2, 0)
        #
        # plt.subplot(1, 3, 1)
        # plt.axis('off')
        # plt.imshow(image_crop)
        #
        # plt.scatter(pose_uvd_relatived_ortho.cpu().detach().numpy()[0, :, 0],
        #             pose_uvd_relatived_ortho.cpu().detach().numpy()[0, :, 1],
        #             c='r')
        #
        # plt.subplot(1, 3, 2)
        # plt.axis('off')
        # plt.imshow(image_crop)
        #
        # plt.subplot(1, 3, 3)
        # plt.axis('off')
        # plt.imshow(image_crop)
        # plt.scatter(vertices_uvd_relatived_ortho.cpu().detach().numpy()[0, :, 0], vertices_uvd_relatived_ortho.cpu().detach().numpy()[0, :, 1],
        #             c='r')
        # plt.show()

    # for step, (image_crop, crop_center_rgb, crop_size_rgb, K, uvd_root, img) in enumerate(test_synthesis):
    #     batch_image = inv_normalize(image_crop[0])
    #     image_crop = batch_image.cpu().detach().numpy().transpose(1, 2, 0)
    #
    #     batch_image = inv_normalize(img[0])
    #     img = batch_image.cpu().detach().numpy().transpose(1, 2, 0)
    #
    #     plt.subplot(2, 4, 1)
    #     plt.axis('off')
    #     plt.imshow(image_crop)
    #
    #     plt.subplot(2, 4, 2)
    #     plt.axis('off')
    #     plt.imshow(img)
    #
    #     plt.subplot(2, 4, 3)
    #     plt.axis('off')
    #     plt.imshow(image_crop)
    #     plt.scatter(crop_center_rgb.cpu().detach().numpy()[0, :, 0], crop_center_rgb.cpu().detach().numpy()[0, :,1], c='r')
    #
    #     plt.subplot(2, 4, 4)
    #     plt.axis('off')
    #     plt.imshow(img)
    #     plt.scatter(uvd_root.cpu().detach().numpy()[0, :, 0], uvd_root.cpu().detach().numpy()[0, :,1], c='r')
    #
    #     plt.show()


