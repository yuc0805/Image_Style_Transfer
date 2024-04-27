CycleGAN Datasets: https://www.kaggle.com/datasets/suyashdamle/cyclegan
    Leo: We can use Monet2photo dataset :) because I like monet
        Datasets over view:
                monet2photo
        ├── monet2photo
        │   ├── train_A (1072 files - style img)
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   ├── train_B (6287 files - real img)
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   └── train_prompts.json
        |
        |   ├── test_A (121 files - style img)
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   ├── test_B (751 files)
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
│   └── test_prompts.json

We can use monet images as style, and real images as content 

Hi Dr.Chiu, since we do not have much compute, please help trimming the dataset when you are preprocessing them
    - I think train_B would be better to be ~ 2000 files, please help trimming other files according to the original proportions.


GAN based Style Transfer:
1 pix2pix 
Github repo:
https://github.com/phillipi/pix2pix 
Some codes and intro in Chinese:
https://aistudio.baidu.com/projectdetail/1119048 
2 CGAN (感觉没啥用）
Paper:
https://ieeexplore.ieee.org/document/8999068 
3 CycleGAN
Github repo:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 


Baseline Models:
GAN-based: T.B.A (Leo think cyclegan or pix2pix would be perfect, you guys can implement the one which you find easier)
VAE-based: StyTr2 (two ViT encoder)
    https://github.com/diyiiyiii/StyTR-2

    - Original paper used a voting scheme and training loss to evaluate, 
    Leo: I think it can be improve by using FID or IS for evaluation for better comparision
    #TODO: train StyTR-2 on Monet training set, and eval the test set using FID

    Hi Prof. Lian, let me know which metric is the GAN-based method you are planning to use.
    I rememer that cyclegan is using FID, but they did not provide code for it :( 
        check out the repo I found below at metric session


Metrics:
FID (Main metric)
    https://github.com/hukkelas/pytorch-frechet-inception-distance

IS (Inception Score)




