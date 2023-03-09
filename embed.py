# scripts to embed a folder into a file containing its face embeddings

import argparse
import sys
from types import SimpleNamespace as nspace
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from functools import partial
import os

sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
#sys.path.append('./models/')
#sys.path.append('./FACE/')
sys.path.append('./FACE/FaceX-Zoo/test_protocol/')
sys.path.append('./FACE/FaceX-Zoo/')
sys.path.append('../FACE/')
sys.path.append('../FACE/FaceX-Zoo/')
sys.path.append('../FACE/FaceX-Zoo/test_protocol/')
#sys.path.append('../../FACE/')
sys.path.append('../../FACE/FaceX-Zoo/test_protocol/')
sys.path.append('../../FACE/FaceX-Zoo/')
#sys.path.append('../../../FACE/')
sys.path.append('../../../FACE/FaceX-Zoo/test_protocol/')
sys.path.append('../../../FACE/FaceX-Zoo/')
sys.path.append('./FACE/models/')
sys.path.append('~/FACE/models/')
sys.path.append('../../FACE/models/')
sys.path.append('../../../FACE/models/')


import test_protocol


augment = T.Compose([T.RandomRotation(5),
                     T.RandomHorizontalFlip(),
                     T.RandomResizedCrop(112, scale=(0.9, 1.0))])

centercrop = lambda n: T.Compose([T.Resize(n), T.CenterCrop(n)])


class FaceEmbedder(nn.Module):
    def __init__(self, core, device, size=112, normalize=False):
        super().__init__()
        self.device = device
        self.size = size
        self.core = core.to(device).eval()
        self.normalize = normalize
        
    def forward(self, imgt, **kwargs):
        imgt = centercrop(self.size)(imgt).to(self.device)
        if self.normalize:
            imgt = 2*imgt - 1
        return self.core(imgt)
        
        
    def compute_embeddings(self, imgs, batch_size=32, **kwargs):
        # batch by batch
#         print(imgs)
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        # to tensor ?
        if isinstance(imgs[0], Image.Image):
            imgt = torch.stack([T.ToTensor()(im) for im in imgs])
            
        out = torch.cat([self.forward(imgt[i:i+batch_size], **kwargs).cpu().detach() for i in tqdm(range(0, len(imgs), batch_size))])
        return out
        

class MagFace(FaceEmbedder):
    def __init__(self, device='cpu'):
        sys.path.extend(['/private/home/mzameshina/FACE/MagFace', 'taming-transformers'])
        from MagFace.inference.network_inf import builder_inf   
        core = builder_inf(nspace(arch='iresnet100',
                  embedding_size=512,
                  resume='/private/home/mzameshina/FACE/models/magface_epoch_00025.pth',
                 cpu_mode=True))
        
        super().__init__(core=core, device=device, size=112)        
        

class FaceNet(FaceEmbedder):
    def __init__(self, device='cpu'):
        from facenet_pytorch import MTCNN, InceptionResnetV1
        
        self.device = device
        core = InceptionResnetV1(pretrained='vggface2') # on gpu
        self.boxes = None
        super().__init__(core=core, device=device, size=160, normalize=True)
        self.mtcnn = MTCNN() # on cpu
        self.image_id = None # for not recomputing boxes
        
        
    def forward(self, imgt, image_id=None):
            
        # do we need to recompute boxes ?
        if image_id is None or image_id != self.image_id:
            self.image_id = image_id
            imgs_for_boxes = [T.ToPILImage()(im) for im in imgt]
            self.boxes = self.mtcnn.detect(imgs_for_boxes)[0]
            
        # differentiable crop with the boxes !
        faces = []
        for im, box in zip(imgt, self.boxes):
            if box is None or True:
                faces.append(im)
            else:
                box = box[0] # only first box
                face = TF.crop(im, int(box[1]), max(0,int(box[0])), 
                           int(box[3]-box[1]), int(box[2]-box[0]))
                faces.append(face)
#                 print('size of face :', face.size(), int(box[1]), int(box[0]), int(box[3]-box[1]), int(box[2]-box[0]))
        faces = torch.stack([centercrop(self.size)(face) for face in faces])
        faces = (2*faces-1).to(self.device)
        return self.core(faces)

class SphereFace(FaceEmbedder):
    def __init__(self, device='cpu'):
        from sphereface_pytorch import net_sphere

        core = getattr(net_sphere,'sphere20a')()
        core.load_state_dict(torch.load('models/sphere20a_20171020.pth', map_location='cpu'))
        core.feature = True
        super().__init__(core=core, device=device, size=(112, 96), normalize=True)


class ArcFace(FaceEmbedder):
    def __init__(self, device='cpu'):
        from arcface_pytorch.models import resnet_face18

        core = resnet_face18(False)
        ckpt = {k[len('module.'):]:v for k, v in torch.load('/private/home/mzameshina/FACE/models/resnet18_110.pth', map_location='cpu').items()}
        core.load_state_dict(ckpt)
        super().__init__(core=core, device=device, size=128)
    
    def forward(self, imgt, **kwargs):
        return FaceEmbedder.forward(self,
                                    imgt = imgt.mean(dim=1, keepdim=True))



class FaceX_Zoo(FaceEmbedder):
    def __init__(self, model_ckpt, device='cpu'):
        import sys
        sys.path.append('FaceX-Zoo')
        from test_protocol.utils.extractor.feature_extractor import CommonExtractor
        from data_processor.test_dataset import CommonTestDataset
        from backbone.backbone_def import BackboneFactory
        from test_protocol.utils.model_loader import ModelLoader
        bkbn = model_ckpt.split('_')[0]
        backbone_factory = BackboneFactory(bkbn, 'FaceX-Zoo/test_protocol/backbone_conf.yaml')
        model_loader = ModelLoader(backbone_factory)
        model_path = f'FaceX-Zoo/models/{model_ckpt}.pt'
        core = model_loader.load_model(model_path).module.to(device)
        super().__init__(core=core, device=device, size=112, normalize=True)
    
    def forward(self, imgt, **kwargs):
        imgt = centercrop(self.size)(imgt).to(self.device)
        imgt = imgt[:, [2, 1, 0]]
        if self.normalize:
            imgt = 2*imgt - 1
        return self.core(imgt)
    
    
class_map = dict(
    facenet=FaceNet,
    sphereface=SphereFace,
    arcface=ArcFace,
    magface=MagFace,
    facexmobile=partial(FaceX_Zoo, 'MobileFaceNet'),
    facexrn50=partial(FaceX_Zoo, 'ResNet_152irse'),
    #facexrn152=partial(FaceX_Zoo, 'ResNet_50ir'),   
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval')
    parser.add_argument('-m', '--method', type=str, default='all', help='face embedding method to use')
    parser.add_argument('-m1', '--method_1', type=str, default=None, help='face embedding method to use')
    parser.add_argument('-f', '--folder', type=str, default='output/closer', help='Folder of images.')
    parser.add_argument('-o', '--output', type=str, default='same', help='Folder of images.')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='device')
    args = parser.parse_args()
    
    paths = [x for x in Path(args.folder).iterdir() if x.name.split('.')[-1].lower() in ('jpeg', 'jpg', 'png')]
#     paths = [x for x in Path(args.folder).iterdir() if x.name.split('.')[-1].lower() in ('jpeg', 'jpg', 'png') and 'cloaked' in x.name]
    try:
        paths = list(sorted(paths, key=lambda x:int(x.name.split('.')[0].split('_')[0])))
    except:
        paths = list(sorted(paths))
    imgs = [Image.open(p) for p in paths]
    
    methods = [args.method, args.method_1]
    if args.method_1 == None:
        methods = [args.method]
    if args.method == 'all':
        methods = class_map.keys()
     
    embeddings = []
    
    for m in methods:
        embedder = class_map[m]()
        embeddings = embedder.compute_embeddings(imgs)
    
        torch.save(embeddings, f'{args.folder}_{m}.pt')