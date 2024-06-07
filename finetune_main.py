import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
#from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR  as StyTR 
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--freeze_part', type=str, default='decoder')
parser.add_argument('--content_dir', default='./datasets/train2014', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./datasets/Images', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--remark',default='freeze_enc_150000',type=str,
                    help="model remark")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

save_dir = os.path.join(args.save_dir,args.remark)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# if not os.path.exists(args.log_dir):
#     os.mkdir(args.log_dir)
# writer = SummaryWriter(log_dir=args.log_dir)

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()

from collections import OrderedDict

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)


for name, param in Trans.named_parameters():
    if name.startswith('encoder'):
        param.requires_grad = True


for name, param in Trans.named_parameters():
    if name.startswith(args.freeze_part):
        param.requires_grad = False


with torch.no_grad():
    network = StyTR.StyTrans(vgg,decoder,embedding, Trans,args)
network.train()

network.to(device)
#network = nn.DataParallel(network, device_ids=[0,1])
content_tf = train_transform()
style_tf = train_transform()



content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
 

# optimizer = torch.optim.Adam([ 
#                               {'params': network.module.transformer.parameters()},
#                               {'params': network.module.decode.parameters()},
#                               {'params': network.module.embedding.parameters()},        
#                               ], lr=args.lr)

optimizer = torch.optim.Adam([ 
                              {'params': network.transformer.parameters()},
                              {'params': network.decode.parameters()},
                              {'params': network.embedding.parameters()},        
                              ], lr=args.lr)


if not os.path.exists(save_dir+"/test"):
    os.makedirs(save_dir+"/test")



for i in tqdm(range(args.max_iter)):

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)  
    out, loss_c, loss_s,l_identity1, l_identity2 = network(content_images, style_images)

    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
                        save_dir, str(i),".jpg"
                    )
        out = torch.cat((content_images,out),0)
        out = torch.cat((style_images,out),0)
        save_image(out, output_name)

        
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1) 
  
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy()
              )
       
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()
    
    # print('gradient check')
    # for name, param in network.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} has gradient")
    #     else:
    #         print(f"{name} has NO gradient")
    # writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    # writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    # writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    # writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        #state_dict = network.module.transformer.state_dict()
        state_dict = network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(save_dir,
                                                           i + 1))

        #state_dict = network.module.decode.state_dict()
        state_dict = network.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(save_dir,
                                                           i + 1))
        #state_dict = network.module.embedding.state_dict()
        state_dict = network.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(save_dir,
                                                           i + 1))

                                                    
#writer.close()

# python train.py --style_dir monet2photo/testA/ --content_dir monet2photo/testB/ --save_dir models/ --batch_size 2 --n_threads 0

# python finetune_main.py --style_dir monet2photo/trainA/  --content_dir monet2photo/trainB_trimed/  --save_dir models/ --batch_size 2 --n_threads 0 --Trans_path experiments/transformer_iter_160000.pth --freeze_part decoder --remark freeze_decoder --vgg experiments/vgg_normalised.pth