import re
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os


def complete_record(start_idx, stat):
    before = [0] * ((start_idx-1000)//1000)
    after = [stat[-1]] * (300-(start_idx-1000)//1000-len(stat))
    return before + stat + after


def parse_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as fp:
        contents = fp.readlines()
        # print(contents)
        line_idx = 0
        epoch_list = []
        psnr_list = []
        loss_list = []
        mse_list = []
        kld_list = []
        psnr_epoch_list = []
        kl_anneal_cyclical = re.findall(r'kl_anneal_cyclical=([T|F])', contents[0])
        kl_anneal_cyclical = True if kl_anneal_cyclical[0]=='T' else False
        kl_anneal_cycle = int(re.findall(r'kl_anneal_cycle=(\d+)', contents[0])[0])
        tfr = float(re.findall(r'tfr=(\d+\.\d+)', contents[0])[0])
        tfr_lower_bound = float(re.findall(r'tfr_lower_bound=(\d+\.\d+)', contents[0])[0])
        tfr_start_decay_epoch = int(re.findall(r'tfr_start_decay_epoch=(\d+)', contents[0])[0])
        niter = int(re.findall(r'niter=(\d+)', contents[0])[0])
        print(tfr)
        print(tfr_lower_bound)
        print(tfr_start_decay_epoch)
        print(niter)
        while line_idx < len(contents):
            # print(contents[line_idx])
            epoch = re.findall(r'epoch: (\d+)', contents[line_idx])
            loss = re.findall(r'\] loss: (\d+.\d+)', contents[line_idx])
            mse = re.findall(r'mse loss: (\d+\.\d+)', contents[line_idx])
            kld = re.findall(r'kld loss: (\d+\.\d+)', contents[line_idx])
            psnr = re.findall(r'validate psnr = (\d+\.\d+)', contents[line_idx])
            if len(psnr):
                psnr_epoch_list.append(epoch_list[-1])
                psnr_list.append(float(psnr[0]))
            if len(epoch):
                epoch_list.append(int(epoch[0]))
                loss_list.append(float(loss[0]))
                mse_list.append(float(mse[0]))
                kld_list.append(float(kld[0]))
            line_idx+=1
        
        
        return psnr_epoch_list, psnr_list
        fig, ax1 = plt.subplots()
        plt.title('Training loss/ratio curve')
        plt.xlabel('epochs')
        ax2 = ax1.twinx()
        ax1.set_ylabel('loss/psnr')
        ax1.plot(epoch_list, kld_list, 'b', label='kld')
        # ax1.plot(epoch_list, loss_list, 'ro', label='total loss')
        ax1.plot(epoch_list, mse_list,'r', label='mse')
        ax1.plot(psnr_epoch_list, psnr_list, 'g.', label='psnr')
        ax1.legend()
        ax1.set_ylim([0.0, 30.0])
        ax2.set_ylabel('ratio')
        ax2.plot(epoch_list, tfr_list, 'm--', label='Teacher ratio')
        ax2.plot(epoch_list, L, '--', color='orange', label='KL weight')
        fig.tight_layout()
        ax2.legend()
        # plt.show()
        plt.savefig(txt_path[:-4]+'.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt1", help="path of log file")
    parser.add_argument("--txt2", help="path of log file")
    args = parser.parse_args()
    epoch, psnr1 = parse_txt(args.txt1)
    epoch, psnr2 = parse_txt(args.txt2)
    plt.figure()
    plt.plot(epoch, psnr1)
    plt.plot(epoch, psnr2)
    plt.legend(['fixed prior', 'learned prior'])
    plt.title('PSNR of fixed/learned prior')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.savefig('fp_lp.png')
    