import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
import collections
import time
from utils.deepsmote_utils import Encoder, Decoder, G_SM1
from config.config import get_args
from dataloader.imbalance_deepsmote import cifar10_deepsmote, cifar100_deepsmote, svhn10_deepsmote, cinic10_deepsmote, tiny200_deepsmote

print(torch.version.cuda) #10.1
t3 = time.time()

def train_deepsmote_generative_model(args, dec_x, dec_y, class_num):
    encoder = Encoder(args)
    decoder = Decoder(args)

    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    #decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    print('train imgs before reshape ',dec_x.shape)
    print('train labels ',dec_y.shape) 
    print(collections.Counter(dec_y))
    dec_x = dec_x.reshape(dec_x.shape[0],3,32,32)
    print('train imgs after reshape ',dec_x.shape)

    #torch.Tensor returns float so if want long then use torch.tensor
    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y,dtype=torch.long)
    cifar_bal = TensorDataset(tensor_x,tensor_y) 
    train_loader = torch.utils.data.DataLoader(cifar_bal, batch_size=args.batch_size, shuffle=True, num_workers=0)

    best_loss = np.inf

    if args.train:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr = args.lr)
        dec_optim = torch.optim.Adam(decoder.parameters(), lr = args.lr)

        for epoch in range(args.epochs):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0
            # train for one epoch -- set nets to train mode
            encoder.train()
            decoder.train()
        
            for images,labs in train_loader:
                # zero gradients for each batch
                encoder.zero_grad()
                decoder.zero_grad()
                # print(images)
                images, labs = images.to(device), labs.to(device)
                # print('images ',images.size()) 
                labsn = labs.detach().cpu().numpy()
                # print('labsn ',labsn.shape, labsn)
                z_hat = encoder(images)
                # print('zhat ', z_hat.size())       
                x_hat = decoder(z_hat) #decoder outputs tanh
                # print('xhat ', x_hat.size())
                mse = criterion(x_hat,images)
                # print('mse ',mse)                  
            
                tc = np.random.choice(class_num,1)
                #tc = 9
                xbeg = dec_x[dec_y == tc]
                ybeg = dec_y[dec_y == tc] 
                xlen = len(xbeg)
                nsamp = min(xlen, 100)
                ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)
                xclass = xbeg[ind]
            
                xclen = len(xclass)
                #print('xclen ',xclen)
                xcminus = np.arange(1,xclen)
                #print('minus ',xcminus.shape,xcminus)            
                xcplus = np.append(xcminus,0)
                #print('xcplus ',xcplus)
                xcnew = (xclass[[xcplus],:])
                #xcnew = np.squeeze(xcnew)
                xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
                #print('xcnew ',xcnew.shape)
            
                xcnew = torch.Tensor(xcnew)
                xcnew = xcnew.to(device)
            
                #encode xclass to feature space
                xclass = torch.Tensor(xclass)
                xclass = xclass.to(device)
                xclass = encoder(xclass)
                #print('xclass ',xclass.shape) 
            
                xclass = xclass.detach().cpu().numpy()        
                xc_enc = (xclass[[xcplus],:])
                xc_enc = np.squeeze(xc_enc)
                #print('xc enc ',xc_enc.shape)
            
                xc_enc = torch.Tensor(xc_enc)
                xc_enc = xc_enc.to(device)
                
                ximg = decoder(xc_enc)
                mse2 = criterion(ximg,xcnew)
                comb_loss = mse2 + mse
                comb_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
            
                train_loss += comb_loss.item()*images.size(0)
                tmse_loss += mse.item()*images.size(0)
                tdiscr_loss += mse2.item()*images.size(0)
                            
            # print avg training statistics 
            train_loss = train_loss/len(train_loader)
            tmse_loss = tmse_loss/len(train_loader)
            tdiscr_loss = tdiscr_loss/len(train_loader)
            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch, train_loss,tmse_loss,tdiscr_loss))
            
            #store the best encoder and decoder models
            if train_loss < best_loss:
                print('Saving..')
                path_enc = '../deepsmote_models/' + args.dataset + '/bst_enc_' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '.pth'
                path_dec = '../deepsmote_models/' + args.dataset + '/bst_dec_' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '.pth'
                
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
                best_loss = train_loss
                
        #in addition, store the final model (may not be the best) for informational purposes
        path_enc = '../deepsmote_models/' + args.dataset + '/f_enc_' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '.pth'
        path_dec = '../deepsmote_models/' + args.dataset + '/f_dec_' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '.pth'
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
    
    t1 = time.time()
    print('total time(min): {:.2f}'.format((t1 - t0)/60))             
    t4 = time.time()
    print('final time(min): {:.2f}'.format((t4 - t3)/60))
    torch.cuda.empty_cache()

def generatesamples(args, dec_x, dec_y, class_num, img_num_per_cls):

    t0 = time.time()
    print('train imgs before reshape ',dec_x.shape) #(44993, 3072) 45500, 3072)
    print('train labels ',dec_y.shape) #(44993,) (45500,)

    # dec_x = dec_x.reshape(dec_x.shape[0],1,28,28) #MNIST dataset
    dec_x = dec_x.reshape(dec_x.shape[0],3,32,32)  #CIFAR dataset

    print('decy ',dec_y.shape)
    print(collections.Counter(dec_y))
    print('train imgs after reshape ',dec_x.shape) #(45000,3,32,32)

    #path on the computer where the models are stored
    encf = []
    decf = []
    enc = '../deepsmote_models/' + args.dataset + '/bst_enc_' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '.pth'
    dec = '../deepsmote_models/' + args.dataset + '/bst_dec_' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '.pth'
    encf.append(enc)
    decf.append(dec)

    #generate some images 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path_enc = encf[0]
    path_dec = decf[0]

    encoder = Encoder(args)
    encoder.load_state_dict(torch.load(path_enc), strict=False)
    encoder = encoder.to(device)

    decoder = Decoder(args)
    decoder.load_state_dict(torch.load(path_dec), strict=False)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    print(img_num_per_cls)

    resx = []
    resy = []

    """functions to create SMOTE images"""
    def biased_get_class1(c):    
        xbeg = dec_x[dec_y == c]
        ybeg = dec_y[dec_y == c]
        
        return xbeg, ybeg #return xclass, yclass

    for i in range(1,class_num):
        xclass, yclass = biased_get_class1(i)
        print(xclass.shape) #(500, 3, 32, 32)
        print(yclass[0]) #(500,)
            
        #encode xclass to feature space
        xclass = torch.Tensor(xclass)
        xclass = xclass.to(device)
        xclass = encoder(xclass)
        print(xclass.shape) #torch.Size([500, 600])
            
        xclass = xclass.detach().cpu().numpy()
        n = img_num_per_cls[0] - img_num_per_cls[i]
        xsamp, ysamp = G_SM1(xclass,yclass,n,i)
        print(xsamp.shape) #(4500, 600)
        print(len(ysamp)) #4500
        ysamp = np.array(ysamp)
        print(ysamp.shape) #4500   

        """to generate samples for resnet"""   
        xsamp = torch.Tensor(xsamp)
        xsamp = xsamp.to(device) #xsamp = xsamp.view(xsamp.size()[0], xsamp.size()[1], 1, 1)
        #print(xsamp.size()) #torch.Size([10, 600, 1, 1])
        ximg = decoder(xsamp)

        ximn = ximg.detach().cpu().numpy()
        print(ximn.shape) #(4500, 3, 32, 32)
        #ximn = np.expand_dims(ximn,axis=1)
        print(ximn.shape) #(4500, 3, 32, 32)
        resx.append(ximn)
        resy.append(ysamp)
    
    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)
    #print(resx1.shape) #(34720, 3, 32, 32)
    #resx1 = np.squeeze(resx1)
    print(resx1.shape) #(34720, 3, 32, 32)
    print(resy1.shape) #(34720,)

    resx1 = resx1.reshape(resx1.shape[0],-1)
    print(resx1.shape) #(34720, 3072)

    dec_x1 = dec_x.reshape(dec_x.shape[0],-1)
    print('decx1 ',dec_x1.shape)
    combx = np.vstack((resx1,dec_x1))
    comby = np.hstack((resy1,dec_y))

    print(combx.shape) #(45000, 3, 32, 32)
    print(comby.shape) #(45000,)

    ifile = '../deepsmote_models/' + args.dataset + '/' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '_train_data.txt'
    np.savetxt(ifile, combx)

    lfile = '../deepsmote_models/' + args.dataset + '/' + args.dataset + '_' + args.imb_type + '_' + 'R' + str(int(1/args.imb_factor)) + '_train_label.txt'
    np.savetxt(lfile,comby) 
    print()

    t1 = time.time()
    print('final time(min): {:.2f}'.format((t1 - t0)/60))

def main():
    """args for AE"""
    args = get_args()
    # The number of total classes in each dataset
    if args.dataset == 'cifar10':
        class_num = 10
        # Loader the imbalance data and number sample pf peer class
        dec_x, dec_y, img_num_per_cls = cifar10_deepsmote(args.imb_type, args.imb_factor)
        print(img_num_per_cls)
    elif args.dataset == 'cifar100':
        class_num = 100
        # Loader the imbalance data and number sample pf peer class
        dec_x, dec_y, img_num_per_cls = cifar100_deepsmote(args.imb_type, args.imb_factor)
        print(img_num_per_cls)
    elif args.dataset == 'svhn10':
        class_num = 10
        # Loader the imbalance data and number sample pf peer class
        dec_x, dec_y, img_num_per_cls = svhn10_deepsmote(args.imb_type, args.imb_factor)
        print(img_num_per_cls)
    elif args.dataset == 'cinic10':
        class_num = 10
        dec_x, dec_y, img_num_per_cls = cinic10_deepsmote(args.imb_type, args.imb_factor)
    elif args.dataset == 'tiny200':
        class_num = 200
        dec_x, dec_y, img_num_per_cls = tiny200_deepsmote(args.imb_type, args.imb_factor)

    # Train DeepSMOTE model for generative samples
    train_deepsmote_generative_model(args, dec_x, dec_y,class_num)

    # Use the best DeepSMOTE model to generate synthetic samples
    generatesamples(args, dec_x, dec_y, class_num, img_num_per_cls)

if __name__ == "__main__":
    main()