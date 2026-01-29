import os
from typing import OrderedDict
import clip
import torch
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, Food101, EMNIST, CelebA
from tqdm import tqdm
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F
from torchvision.utils import save_image
import PIL
from sklearn.decomposition import FastICA
from torch.nn.utils import clip_grad_norm_

import warnings
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch import nn

from scipy.stats import ortho_group
from typing import Union
from typing_extensions import Literal
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def construct_invertible_mlp(
    n: int = 20,
    n_layers: int = 2,
    n_iter_cond_thresh: int = 10000,
    cond_thresh_ratio: float = 0.25,
    weight_matrix_init: Union[Literal["pcl"], Literal["rvs"]] = "pcl",
    act_fct: Union[
        Literal["relu"],
        Literal["leaky_relu"],
        Literal["elu"],
        Literal["smooth_leaky_relu"],
        Literal["softplus"],
    ] = "leaky_relu",
):
    """
    Create an (approximately) invertible mixing network based on an MLP.
    Based on the mixing code by Hyvarinen et al.

    Args:
        n: Dimensionality of the input and output data
        n_layers: Number of layers in the MLP.
        n_iter_cond_thresh: How many random matrices to use as a pool to find weights.
        cond_thresh_ratio: Relative threshold how much the invertibility
            (based on the condition number) can be violated in each layer.
        weight_matrix_init: How to initialize the weight matrices.
        act_fct: Activation function for hidden layers.
    """

    class SmoothLeakyReLU(nn.Module):
        def __init__(self, alpha=0.2):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return self.alpha * x + (1 - self.alpha) * torch.log(1 + torch.exp(x))

    def get_act_fct(act_fct):
        if act_fct == "relu":
            return torch.nn.ReLU, {}, 1
        if act_fct == "leaky_relu":
            return torch.nn.LeakyReLU, {"negative_slope": 0.2}, 1
        elif act_fct == "elu":
            return torch.nn.ELU, {"alpha": 1.0}, 1
        elif act_fct == "max_out":
            raise NotImplemented()
        elif act_fct == "smooth_leaky_relu":
            return SmoothLeakyReLU, {"alpha": 0.2}, 1
        elif act_fct == "softplus":
            return torch.nn.Softplus, {"beta": 1}, 1
        else:
            raise Exception(f"activation function {act_fct} not defined.")

    layers = []
    act_fct, act_kwargs, act_fac = get_act_fct(act_fct)

    # Subfuction to normalize mixing matrix
    def l2_normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat * Amat, axis))
        Amat = Amat / l2norm
        return Amat

    condList = np.zeros([n_iter_cond_thresh])
    if weight_matrix_init == "pcl":
        for i in range(n_iter_cond_thresh):
            A = np.random.uniform(-1, 1, [n, n])
            A = l2_normalize(A, axis=0)
            condList[i] = np.linalg.cond(A)
        condList.sort()  # Ascending order
    condThresh = condList[int(n_iter_cond_thresh * cond_thresh_ratio)]
    print("condition number threshold: {0:f}".format(condThresh))

    for i in range(n_layers):

        lin_layer = nn.Linear(n, n, bias=False)

        if weight_matrix_init == "pcl":
            condA = condThresh + 1
            while condA > condThresh:
                weight_matrix = np.random.uniform(-1, 1, (n, n))
                weight_matrix = l2_normalize(weight_matrix, axis=0)

                condA = np.linalg.cond(weight_matrix)
                # print("    L{0:d}: cond={1:f}".format(i, condA))
            print(
                f"layer {i+1}/{n_layers},  condition number: {np.linalg.cond(weight_matrix)}"
            )
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)

        elif weight_matrix_init == "rvs":
            weight_matrix = ortho_group.rvs(n)
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)
        elif weight_matrix_init == "expand":
            pass
        else:
            raise Exception(f"weight matrix {weight_matrix_init} not implemented")

        layers.append(lin_layer)

        if i < n_layers - 1:
            layers.append(act_fct(**act_kwargs))

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    #for p in mixing_net.parameters():
    #    p.requires_grad = False

    return mixing_net


def get_flow(
    n_in: int = 512,
    n_out: int = 512,
    init_identity: bool = False,
    coupling_block: Union[Literal["gin"], Literal["glow"]] = "gin",
    num_nodes: int = 8,
    node_size_factor: int = 1,
):
    """
    Creates an flow-based network.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        init_identity: Initialize weights to identity network.
        coupling_block: Coupling method to use to combine nodes.
        num_nodes: Depth of the flow network.
        node_size_factor: Multiplier for the hidden units per node.
    """

    # do lazy imports here such that the package is only
    # required if one wants to use the flow mixing
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm

    def _invertible_subnet_fc(c_in, c_out, init_identity):
        subnet = nn.Sequential(
            nn.Linear(c_in, c_in * node_size_factor),
            nn.ReLU(),
            #nn.Linear(c_in * node_size_factor, c_in * node_size_factor),
            #nn.ReLU(),
            nn.Linear(c_in * node_size_factor, c_out),
        )
        if init_identity:
            subnet[-1].weight.data.fill_(0.0)
            subnet[-1].bias.data.fill_(0.0)
        return subnet

    assert n_in == n_out

    if coupling_block == "gin":
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == "glow"
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(n_in, name="input")]

    for k in range(num_nodes):
        nodes.append(
            Ff.Node(
                nodes[-1],
                block,
                {
                    "subnet_constructor": lambda c_in, c_out: _invertible_subnet_fc(
                        c_in, c_out, init_identity
                    ),
                    "clamp": 2.0,
                },
                name=f"coupling_{k}",
            )
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)



class CLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair and one negative pair"""

    @abstractmethod
    def loss(self, z1, z2_con_z1):
        """
        z1_t = h(z1)
        z2_t = h(z2)
        z3_t = h(z3)
        and z1 ~ p(z1), z3 ~ p(z3)
        and z2 ~ p(z2 | z1)

        returns the total loss and componentwise contributions
        """
        pass

    def __call__(self, z1, z2_con_z1):
        return self.loss(z1, z2_con_z1)

def _logmeanexp(x, dim):
    # do the -log thing to use logsumexp to calculate the mean and not the sum
    # as log sum_j exp(x_j - log N) = log sim_j exp(x_j)/N = log mean(exp(x_j)
    N = torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
    return torch.logsumexp(x, dim=dim) - torch.log(N)

class LpCLIPLoss(CLLoss):
    """Extended InfoNCE objective for non-normalized representations based on an Lp norm.

    Args:
        p: Exponent of the norm to use.
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
        simclr_compatibility_mode: Use logsumexp (as used in SimCLR loss) instead of logmeanexp
        pow: Use p-th power of Lp norm instead of Lp norm.
    """

    def __init__(
        self,
        p: int,
        tau: float = 1.0,
        alpha: float = 0.5,
        simclr_compatibility_mode: bool = False,
        pow: bool = True,
    ):
        self.p = p
        self.tau = tau
        self.alpha = alpha
        self.simclr_compatibility_mode = simclr_compatibility_mode
        self.pow = pow

    def loss(self, z1_rec, z2_con_z1_rec):
       
        z3_rec = torch.roll(z2_con_z1_rec, 1, 0)
        z4_rec = torch.roll(z1_rec, 1, 0)
        if self.p < 1.0:
            # add small epsilon to make calculation of norm numerically more stable
            neg = torch.norm(
                torch.abs(z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1) + 1e-12),
                p=self.p,
                dim=-1,
            )
            pos = torch.norm(
                torch.abs(z1_rec - z2_con_z1_rec) + 1e-12, p=self.p, dim=-1
            )
        else:
            # TODO: verify this
            # neg = torch.norm(z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1), p=self.p, dim=-1)
            # pos = torch.norm(z1_rec - z2_con_z1_rec, p=self.p, dim=-1)
            neg = torch.norm(
                z1_rec.unsqueeze(1) - z3_rec.unsqueeze(0), p=self.p, dim=-1
            )
            pos = torch.norm(z1_rec - z2_con_z1_rec, p=self.p, dim=-1)

        if self.pow:
            neg = neg.pow(self.p)
            pos = pos.pow(self.p)

        # all = torch.cat((neg, pos.unsqueeze(1)), dim=1)

        if self.simclr_compatibility_mode:
            neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

            loss_pos = pos / self.tau
            loss_neg = torch.logsumexp(-neg_and_pos / self.tau, dim=1)
        else:
            loss_pos = pos / self.tau
            loss_neg = _logmeanexp(-neg / self.tau, dim=1)

        loss = 2 * (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)
        #loss_std = torch.std(loss)

        loss_pos_mean = torch.mean(loss_pos)
        loss_neg_mean = torch.mean(loss_neg)

        # construct symmetry loss

        if self.p < 1.0:
            # add small epsilon to make calculation of norm numerically more stable
            neg_sym = torch.norm(
                torch.abs(z2_con_z1_rec.unsqueeze(0) - z4_rec.unsqueeze(1) + 1e-12),
                p=self.p,
                dim=-1,
            )
            pos_sym = torch.norm(
                torch.abs(z2_con_z1_rec-z1_rec) + 1e-12, p=self.p, dim=-1
            )
        else:
            # TODO: verify this
            # neg = torch.norm(z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1), p=self.p, dim=-1)
            # pos = torch.norm(z1_rec - z2_con_z1_rec, p=self.p, dim=-1)
            neg_sym = torch.norm(
                z2_con_z1_rec.unsqueeze(1) - z4_rec.unsqueeze(0), p=self.p, dim=-1
            )
            pos_sym = torch.norm(z2_con_z1_rec - z1_rec, p=self.p, dim=-1)

        if self.pow:
            neg_sym = neg_sym.pow(self.p)
            pos_sym = pos_sym.pow(self.p)

        # all = torch.cat((neg, pos.unsqueeze(1)), dim=1)

        if self.simclr_compatibility_mode:
            neg_and_pos_sym = torch.cat((neg_sym, pos_sym.unsqueeze(1)), dim=1)

            loss_pos_sym = pos_sym / self.tau
            loss_neg_sym = torch.logsumexp(-neg_and_pos_sym / self.tau, dim=1)
        else:
            loss_pos_sym = pos_sym / self.tau
            loss_neg_sym = _logmeanexp(-neg_sym / self.tau, dim=1)

        loss_sym= 2 * (self.alpha * loss_pos_sym + (1.0 - self.alpha) * loss_neg_sym)

        loss_mean_sym = torch.mean(loss_sym)
        #loss_std = torch.std(loss)

        loss_pos_mean_sym = torch.mean(loss_pos_sym)
        loss_neg_mean_sym = torch.mean(loss_neg_sym)


        # print(loss_std)

        return (loss_mean + loss_mean_sym)/2.


class CELEBATEXT(torch.utils.data.Dataset):
    """Load dataset"""
    def __init__(self, root='/home/yuhang/CLIP-main/dataset/celeba', device='cpu', transform = None):
        super(CELEBATEXT, self).__init__()
         
        self.device = device
        self.root = root
        self.base_img = 'img_align_celeba'
        self.base_text = 'celeba_caption'
        self.filename  = os.listdir(os.path.join(self.root, self.base_img))
        self.filename.sort()
        self.filename_text  = os.listdir(os.path.join(self.root, self.base_text))
        self.filename_text.sort()

        self.len = len(self.filename)
        self.lenq = len(self.filename_text)
        self.transform  = transform


     
        self.texts = []
        for item in range(0, self.len): #self.len-1
            text = Path(os.path.join(self.root, "celeba_caption", self.filename_text[item])).read_text()#list(np.loadtxt(os.path.join(self.root, "celeba_caption", self.filename_text[item]),dtype=str)).join()
            self.texts.append(text)
        self.tokens =  clip.tokenize(self.texts)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))
        
        token = self.tokens[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, token








class CELEBADataset(torch.utils.data.Dataset):
    """Load dataset"""
    def __init__(self, data='/home/yuhang/CLIP-main/dataset/celeba', device='cpu'):
        super(CELEBADataset, self).__init__()
         
        self.device = device

        self.latent_classes = np.load(os.path.join(data, "celeba_latent.npy")).astype(np.float32)
        self.img = np.load(os.path.join(data, "celeba_img.npy")).astype(np.float32)
        self.latent_text = np.load(os.path.join(data, "celeba_latent_text.npy")).astype(np.float32)
        self.latent_ica = np.load(os.path.join(data, "celeba_latent_fastica_64.npy")).astype(np.float32)

        self.x = torch.from_numpy(self.img)  # .to(device)
        self.latent = torch.from_numpy(self.latent_classes)  # .to(device)
        self.latent_t = torch.from_numpy(self.latent_text)  # .to(device)
        self.latent_fastica = torch.from_numpy(self.latent_ica)  # .to(device)

        self.len = self.x.shape[0]
        self.latent_dim = self.latent.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return self.x[index], self.latent[index], self.latent_t[index], self.latent_fastica[index]




# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
a = clip.available_models()

# Load the dataset
root = os.path.expanduser("/home/yuhang/CLIP-main/dataset/")

#train_dataset = CelebA(root, download=True,  split='all', transform=preprocess)
#test_dataset = CelebA(root, download=True,  split='all', transform=preprocess)

# train_dataset = CELEBADataset()
# test_dataset = CELEBADataset()

imagesize = 64
ResizeOrginal = transforms.Resize(size=imagesize)




class ConvDecoder(nn.Module):
    def __init__(self, input_dim=512):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 3, 4, 2, 1) # 3* 64 x 64
        #self.conv_final = nn.Conv2d(32, 1, 5, 1) # 1* 28 x 28

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img



class SoftclipLayer(nn.Module):
    """Normalize the data to a hyperrectangle with fixed/learnable size."""

    def __init__(self, n=512, init_abs_bound=1.0, fixed_abs_bound=False):
        super().__init__()
        self.fixed_abs_bound = fixed_abs_bound
        if fixed_abs_bound:
            self.max_abs_bound = torch.ones(n, requires_grad=False) * init_abs_bound
        else:
            self.max_abs_bound = nn.Parameter(
                torch.ones(n, requires_grad=True) * init_abs_bound
            )

    def forward(self, x):
        x = torch.sigmoid(x)
        x = x * self.max_abs_bound.to(x.device).unsqueeze(0)

        return x

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 512),
        nn.ReLU(),
        nn.Linear(512, dims_out)
        )
class Finetune(nn.Module):
    def __init__(self, input_dim=512):
        super(Finetune, self).__init__()
        iidfeature = 512
        self.mlp_img = Ff.SequenceINN(input_dim)
        for k in range(8):
            self.mlp_img.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

        self.mlp_text = Ff.SequenceINN(input_dim)
        for k in range(8):
            self.mlp_text.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

        self.mlp_img = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, iidfeature),
        )
        self.mlp_text = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, iidfeature),
        )
        self.SoftclipLayer = SoftclipLayer(n=iidfeature)

    def forward(self, img, text):
        imgs = self.mlp_img(img) + img
        texts = self.mlp_text(text) + text
        ft_img = self.SoftclipLayer(imgs)
        tt_text = self.SoftclipLayer(texts)
        return ft_img, tt_text

class NormLayer(nn.Module):
    def __init__(self, input_dim=512):
        super(NormLayer, self).__init__()      
        self.SoftclipLayer = SoftclipLayer(n=input_dim)

    def forward(self, img):
        ft_img = self.SoftclipLayer(img)
        return ft_img



n_disentangle = 512




def extract():
    train_dataset = CELEBATEXT(transform=preprocess)
    test_dataset = CELEBATEXT(transform=preprocess)

    Allx = np.empty((200,3,imagesize,imagesize))
    Allz = np.empty((200,n_disentangle))
    Allt = np.empty((200,n_disentangle))

    for images, labels in tqdm(DataLoader(train_dataset, batch_size=100, shuffle=True),leave=False):
            features = model.encode_image(images.to(device))
            features_text = model.encode_text(labels.to(device))
            
            
            images_origsize = ResizeOrginal(images.to(device))
            feateval32 = features.to(torch.float32)
            feateval32_text = features_text.to(torch.float32)


            Allx = np.append(Allx, images_origsize.cpu().detach().numpy(), axis=0)
            Allz = np.append(Allz, feateval32.cpu().detach().numpy(), axis=0)
            Allt = np.append(Allt, feateval32_text.cpu().detach().numpy(), axis=0)

    x = Allx[200:,:,:,:]
    y = Allz[200:,:]
    t = Allt[200:,:]

    ica = FastICA(n_components=64, random_state=0) 
    feat_ica = ica.fit_transform(y)



    
    np.save('/home/yuhang/CLIP-main/dataset/celeba/celeba_img.npy',x)
    np.save('/home/yuhang/CLIP-main/dataset/celeba/celeba_latent.npy',y)
    np.save('/home/yuhang/CLIP-main/dataset/celeba/celeba_latent_text.npy',t)
    np.save('/home/yuhang/CLIP-main/dataset/celeba/celeba_latent_fastica.npy',feat_ica)





def finetune_clip(epoch):
    model.train()
    train_dataset = CELEBATEXT(transform=preprocess)
    test_dataset = CELEBATEXT(transform=preprocess)
    MCRL = LpCLIPLoss(p=1, tau=1., simclr_compatibility_mode=True)
    
    normlayer = NormLayer()
    ft_clip_optimizer = torch.optim.Adam(list(model.parameters()) + list(normlayer.parameters()), lr=3e-5)

    for images, labels in tqdm(DataLoader(train_dataset, batch_size=100, shuffle=True),leave=False):
            ft_clip_optimizer.zero_grad()

            features = normlayer(model.encode_image(images.to(device)))
            features_text = normlayer(model.encode_text(labels.to(device)))
            
            loss = MCRL(features,features_text)
            loss.backward()
            ft_clip_optimizer.step()
    print('Epoch {} \t loss: {:.4f}'.format(epoch, loss))
    torch.save(ftmodel.state_dict(), '/home/yuhang/CLIP-main/checkpoints/finetune_clip_celeba_p1_unbox.pt')
    torch.save(normlayer.state_dict(), '/home/yuhang/CLIP-main/checkpoints/finetune_clip_celeba_p1_unbox_normlayer.pt')




def transform_ICA():
    y = np.load(os.path.join("/home/yuhang/CLIP-main/dataset/celeba/celeba_latent.npy")).astype(np.float32)

    ica = FastICA(n_components=512, random_state=0)
    
    start_time = time.time()
    feat_val_ica = ica.fit_transform(y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"time is: {elapsed_time} s")
    
    # feateval_ica = feat_val_ica #torch.from_numpy(feat_val_ica).to(device)
    # feateval_ica32 = feateval_ica#.cpu().detach().numpy()
    # mixing = ica.mixing_
    #np.save('/home/yuhang/CLIP-main/dataset/celeba/celeba_latent_fastica_512.npy',feateval_ica32)




def finetune(epoch):
    ftmodel = Finetune().to(device)
    train_dataset = CELEBADataset()
    MCRL = LpCLIPLoss(p=1, tau=1., simclr_compatibility_mode=True)
    ft_optimizer = torch.optim.Adam(ftmodel.parameters(), lr=1e-3)
    params = ftmodel.parameters()
    for images, latent, latent_text,latent_fastica in tqdm(DataLoader(train_dataset, batch_size=128, shuffle=True),leave=False):
            
            
            features = latent.to(device)
            features_text = latent_text.to(device)

            ft_optimizer.zero_grad()
            ft_img, ft_text  = ftmodel(features.to(torch.float32),features_text.to(torch.float32))
            loss = MCRL(ft_img,ft_text) 
            loss.backward()
            ft_optimizer.step()
    print('Epoch {} \t loss: {:.4f}'.format(epoch, loss))
    torch.save(ftmodel.state_dict(), '/home/yuhang/CLIP-main/checkpoints/finetune_same_celeba_p1_box_invert.pt')


Recon_Model= ConvDecoder(input_dim=512).to(device)
ftmodel = Finetune().to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(Recon_Model.parameters(), lr=1e-4)

def train(epoch):
    Recon_Model.train()
    ckpt_file = '/home/yuhang/CLIP-main/checkpoints/finetune_same_celeba_p1_box_invert.pt'
    ftmodel_dict = ftmodel.state_dict()
    state_dict = torch.load(ckpt_file)
    new_state_dict = OrderedDict()
    new_state_dict = {k:v for k,v in state_dict.items() if k in ftmodel_dict}
    
    ftmodel_dict.update(new_state_dict)
    ftmodel.load_state_dict(ftmodel_dict)
    ftmodel.eval()

    train_dataset = CELEBADataset()
   
    for images, latent, latent_text, feature_ica in tqdm(DataLoader(train_dataset, batch_size=100, shuffle=True),leave=False):
            features = latent.to(device)
            features_text = latent_text.to(device)
            ft_img, ft_text = ftmodel(features.to(torch.float32),features_text.to(torch.float32))
            
            images_origsize = ResizeOrginal(images.to(device))
            optimizer.zero_grad()
            recon_img = Recon_Model(ft_img.to(torch.float32))
            loss = criterion(recon_img,images_origsize)
            loss.backward()
            optimizer.step()
    torch.save(Recon_Model.state_dict(), '/home/yuhang/CLIP-main/checkpoints/recon_CELEBA_p1_box_512.pt')
    save_image(recon_img[0], '/home/yuhang/CLIP-main/recon_celeba/reconstructed_image_{}.png'.format(epoch), range = (0,1))
    print('Epoch {} \t loss: {:.4f}'.format(epoch, loss))



def fastica_train(epoch):
    Recon_Model.train()
    train_dataset = CELEBADataset()
   
    for images, latent, latent_text, latent_ica in tqdm(DataLoader(train_dataset, batch_size=100, shuffle=True),leave=False):
            features = latent_ica.to(device)
            images_origsize = ResizeOrginal(images.to(device))
            optimizer.zero_grad()
            recon_img = Recon_Model(features.to(torch.float32))
            loss = criterion(recon_img,images_origsize)
            loss.backward()
            optimizer.step()
    torch.save(Recon_Model.state_dict(), '/home/yuhang/CLIP-main/checkpoints/recon_CELEBA_fastica_64_1e_4_e300.pt')
    save_image(recon_img[0], '/home/yuhang/CLIP-main/recon_celeba/reconstructed_image_{}.png'.format(epoch), range = (0,1))
    print('Epoch {} \t loss: {:.4f}'.format(epoch, loss))



def test(ckpt_file):
    Recon_Model.eval()
    Recon_Model.load_state_dict(torch.load(ckpt_file))

    # Recon_Model_dict = Recon_Model.state_dict()
    
    # state_dict = torch.load(ckpt_file)
    # new_state_dict = OrderedDict()
    # new_state_dict = {k:v for k,v in state_dict.items() if k in Recon_Model_dict}
    
    # Recon_Model_dict.update(new_state_dict)
    # Recon_Model.load_state_dict(Recon_Model_dict)
    train_dataset = CELEBADataset()
    # test_dataset = CELEBADataset()

    for images, latent, latent_text, latent_ica in tqdm(DataLoader(train_dataset, batch_size=100, shuffle=False),leave=False):
            images_example = images[20:30,:,:,:].to(device)
            break
    features = latent_ica.to(device) #model.encode_image(images.to(device))
    z1_internal = 1e-2*torch.ones(1,1)*torch.linspace(-1.2,1.2,10)
    z1_internal = z1_internal.to(device)
    t=0
    b=64
    index = [[17, 1], [2, 2], [3, 3], [22, 4], [15, 5], [6, 6], [7,7], [23, 8], [19, 9], [30,11]]
    for i in range(b): # 
        #if i >= (b-512):
            for j in range(10):
                #j=0
                features = latent_ica.to(device)
                features[:,i:i+1] =  z1_internal[:,j:j+1] #torch.cat((z1_internal[:,j:j+1], z[:,1:2], z[:,2:]), 1)
                recon_img = Recon_Model(features.to(torch.float32))
                for idx in range(10):
                    save_image(recon_img[index[idx][0]], '/home/yuhang/CLIP-main/intervceleba_fastica_64/recon_image_{}_feature_{}_interven_{}.png'.format(index[idx][1],i, j),  range = (0,1)) 
                #j = 1
                #save_image(recon_img[30], '/home/yuhang/CLIP-main/intervceleba/recon_image_feature_{}_interven_{}.png'.format(i, j),  range = (0,1)) 
                # index: digit, 0: 8, 1: 9, 2:6, 3:3, 6:7, 7:1, 9:0, 13:4, 15:3, 37:2
            t = t + 1
            




def adaptor_test(ckpt_file):
    #Recon_Model.eval()
    Recon_Model.load_state_dict(torch.load(ckpt_file))
    Recon_Model.eval()

    ckpt_file1 = '/home/yuhang/CLIP-main/checkpoints/finetune_same_celeba_p1_box_invert.pt'
    
    ftmodel.load_state_dict(torch.load(ckpt_file1))
    # ftmodel_dict = ftmodel.state_dict()
    # state_dict = torch.load(ckpt_file1)
    # new_state_dict = OrderedDict()
    # new_state_dict = {k:v for k,v in state_dict.items() if k in ftmodel_dict}
    
    # ftmodel_dict.update(new_state_dict)
    # ftmodel.load_state_dict(ftmodel_dict)
    ftmodel.eval()
    
    train_dataset = CELEBADataset()
    # test_dataset = CELEBADataset()

    for images, latent, latent_text, latent_ica in tqdm(DataLoader(train_dataset, batch_size=100, shuffle=False),leave=False):
            images_example = images[20:30,:,:,:].to(device)
            break
    
    features = latent.to(device)
    features_text = latent_text.to(device)
    ft_img, ft_text = ftmodel(features.to(torch.float32),features_text.to(torch.float32))
            



    z1_internal = 1*torch.ones(1,1)*torch.linspace(-0.2,1.2,10)
    z1_internal = z1_internal.to(device)
    t=0
    b=512
    a = ft_img[:,511]
    index = [[17, 1], [4, 2], [3, 3], [23, 4], [15, 5], [6, 6], [7,7], [24, 8], [19, 9], [30,11]]
    for i in range(b): # 
        #if i >= 510:
            for j in range(10):
                #j=0
                features = ft_img.clone() 
                features[:,i:i+1] =  z1_internal[:,j:j+1] #torch.cat((z1_internal[:,j:j+1], z[:,1:2], z[:,2:]), 1)
                recon_img = Recon_Model(features.to(torch.float32))
                for idx in range(10):
                    save_image(recon_img[index[idx][0]], '/home/yuhang/CLIP-main/recon_CELEBA_p1_box_512/recon_image_{}_feature_{}_interven_{}.png'.format(index[idx][1],i, j),  range = (0,1)) 
                #j = 1
                #save_image(recon_img[30], '/home/yuhang/CLIP-main/intervceleba/recon_image_feature_{}_interven_{}.png'.format(i, j),  range = (0,1)) 
                # index: digit, 0: 8, 1: 9, 2:6, 3:3, 6:7, 7:1, 9:0, 13:4, 15:3, 37:2
            t = t + 1
            


# extract()


# for epoch in range(50):
#     finetune(epoch)



# for epoch in range(100):
#    train(epoch)
transform_ICA()
#test(ckpt_file='/home/yuhang/CLIP-main/checkpoints/recon_CELEBA_fastica_64_1e_4_e300.pt') #fastica 512 dim
#adaptor_test(ckpt_file='/home/yuhang/CLIP-main/checkpoints/recon_CELEBA_p1_box_512.pt') #fastica 512 dim