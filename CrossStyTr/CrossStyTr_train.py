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
#import models.transformer as transformer
#import models.StyTR  as StyTR 
import CrossStyTr_model
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
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
parser.add_argument('--content_dir', default='monet2photo/trainB', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='monet2photo/trainA', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=12)
parser.add_argument('--save_model_interval', type=int, default=2000)
parser.add_argument('--remark', type=str, default='reamark')

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# if not os.path.exists(args.log_dir):
#     os.mkdir(args.log_dir)
# writer = SummaryWriter(log_dir=args.log_dir)

vgg = CrossStyTr_model.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = CrossStyTr_model.decoder

with torch.no_grad():
    network = CrossStyTr_model.CrossStyTr(encoder=vgg, decoder=decoder, device=device)
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

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

save_dir = os.path.join(args.save_dir,args.remark)

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
    #print('content images',content_images)
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

    # writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    # writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    # writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    # writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:

            checkpoint = {
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': i + 1
            }
            for key in checkpoint['model_state_dict'].keys():
                checkpoint['model_state_dict'][key] = checkpoint['model_state_dict'][key].to(torch.device('cpu'))

            # Save the checkpoint
            torch.save(checkpoint, '{:s}/checkpoint_{:d}.pth'.format(args.save_dir, i + 1))

                                                    
#writer.close()

#python CrossStyTr/CrossStyTr_train.py --save_dir models/ --batch_size 2 --n_threads 0 --remark dummy_run
# python CrossStyTr/CrossStyTr_train.py --save_dir models/ --batch_size 4 --n_threads 12 --max_iter 160000 --remark c10s7 --style_dir monet2photo/trainA --content_dir monet2photo/trainB_trimed --content_weight 10 --style_weight 7
# python CrossStyTr/CrossStyTr_train.py --save_dir models/ --batch_size 4 --n_threads 12 --max_iter 160000 --remark c10s7 --style_dir monet2photo/trainA --content_dir monet2photo/trainB_trimed --content_weight 10 --style_weight 7