#!/usr/bin/env python


from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from helper_functions import *
from Network import *
from nerf import *
import argparse
import tqdm as tq
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import pry
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def Train(args):
    # writer = SummaryWriter(args.log_path)
    # data = NeRF_dataset(args.BasePath, args.mode)

    # f,P,images = data.tiny_nerf_data(device)

    # input_channels = 99

    # model = NeRF(input_channels, args.width).to(device)

    # optimiser = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate, betas = (0.9,0.9999))

    # start = load_model(model, args)
    # # pry()
    # model.train()

    # for i in tq.tqdm(range(start, args.total_iterations)):
    #     print(f"training for{i} began now")
    #     loss_per_iteration = 0

    #     for j in tq.tqdm(range(P.shape[0])):
    #         image = images[j]
    #         P_c = P[j]

    #         model.train()
    #         H = image.shape[0]
    #         W = image.shape[1]
    #         training_image_ , iamge_ = get_rendering(H,W,f,P_c,model, True, args)
    #         image = image.to(torch.float64)
    #         print(f"traiing iamge: {(training_image_.dtype)}")
    #         print(f"output:{(image.dtype)}")
    #         # images_b = image[iamge_]
    #         loss_for_this_view = loss(image, 
            
    #         )

    #         optimiser.zero_grad()
            
    #         loss_for_this_view.backward()

    #         optimiser.step()

    #         loss_per_iteration += loss_for_this_view


    #         writer.add_scalar("lossEveryview", loss_for_this_view, i*P.shape[0] + i)
    #         writer.flush()

    #     writer.add_scalar("loss_for_this_iteration", loss_per_iteration, i)
    #     writer.flush()
    #     print(f"iteration_no: {i}, loss_for_this_iteration: {loss_per_iteration}\n")


    #     if i % 10 == 0:
    #         if not (os.path.isdir(args.checkpoint_path)):
    #             os.makedirs(args.checkpoint_path)

    #         checkpoint_name = args.checkpoint_path + os.sep + "model" + str(i) + ".ckpt"

    #         torch.save({'Iteration_No': i, 'model_state_dict': model.state_dict(), 'optimiser': optimiser.state_dict(), 'loss': loss_per_iteration}, checkpoint_name)




    # print("........................Done with Training lets test it............................")

def Training(args):
    print("inside the training")
    writer = SummaryWriter(args.log_path)
    data = NeRF_dataset(args.BasePath, args.mode)
    input_channels = 39
    f,P,images = data.tiny_nerf_data(device)
    model = NeRF(input_channels, args.width).to(device)
    # model = model.to(device)
    # optimiser = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate, betas = (0.9,0.9999))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9999))


    torch.manual_seed(1200)
    random.seed(1200)

    loss_ = []
    epochs = []
    psnr = []
    start = load_model(model, args)
    for i in range(start, 1000):
        j_ = i+1
        print(f"inside the epochs{j_} ")
        img_idx = random.randint(0, images.shape[0]-1)
        # print(device)
        
        img = images[img_idx].double().to(device)
        P_c = P[img_idx].double().to(device)
        H = img.shape[0]
        W = img.shape[1]

        training_image_  = get_rendering(H,W,f,P_c,model, True, args)
        # print("done with network")
        loss = F.mse_loss(training_image_.double(), img.double()) #photometric loss
        print(f"loss for {i}th this epoch is {loss}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("lossEveryview", loss, i*P.shape[0] + i)
        writer.flush()

        if i % 5 == 0:
            training_image_  = get_rendering(H,W,f,P_c,model, True, args)
            loss = F.mse_loss(training_image_, img)
            print("Loss", loss.item())
            Loss_ = float(loss.item())
            loss_.append(Loss_)
            epochs.append(i+1)

            Peak_signal_to_noise_ratio = -10 * torch.log10(loss)
            psnr.append(Peak_signal_to_noise_ratio.item())


            psnr_image_name = "./output/psnr_output_new/rendered_image_{}.png".format(i+1)
            output_rendered_image_name = f"./NeRF/NeRF/output/rendered_image_output_new/rendered_image_{i+1}.png"
            checkpoint_name = args.checkpoint_path + os.sep + "model" + str(i+1) + ".ckpt"
            

            plt.imsave(output_rendered_image_name, training_image_.detach().cpu().numpy())

            plt.figure(figsize = (8,4))
            plt.plot(epochs, psnr)
            plt.xlabel("No of Epochs")
            plt.ylabel("Peak_signal_to_noise_ratio")
            plt.title("PSNR VS NO of Epochs")
            plt.savefig(psnr_image_name)

            torch.save({'Epochs': i+1, 'model_state_dict': model.state_dict(), 'optimiser': optimizer.state_dict(), 'loss': Loss_}, checkpoint_name)
    # plot_figures(epochs, Loss)

    SaveName =  args.checkpoint_path + os.sep + "model_final" + str(i+1) + ".ckpt"
                
    torch.save({'i': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()},
            SaveName)  

    plt.plot(epochs,loss_)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("loss trend")
    plt.savefig("loss.png")



def main():
    

    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="./lego-20230307T162216Z-001/lego",
        help="dataset path",
    )
    Parser.add_argument(
        "--checkpoint_path",
        default="./checkpoints/log_l",
        help="checkpoint_path",
    )

    Parser.add_argument(
        "--mode",
        default="Train",
        help="train / test / val",
    )
    Parser.add_argument(
        "--log_path",
        default="./checkpoints/log_l/logs",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    Parser.add_argument(
        "--total_iterations",
        type=int,
        default=300000,
        help="no of iterations for training",
    )
    Parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="number of channels for the network",
    )
    Parser.add_argument(
        "--learning_rate",
        type=int,
        default=0.008,
        help="training data learning rate",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=False,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--no_frequencies",
        type=int,
        default=6,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    
    
    )
    Parser.add_argument(
        "--threshold_near",
        type=int,
        default=2,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    
    )
    Parser.add_argument(
        "--threshold_far",
        type=int,
        default=6,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    
    )
    Parser.add_argument(
        "--sample_num",
        type=int,
        default=64,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    
    )

    Parser.add_argument(
        "--no_of_rays",
        type=int,
        default=3000,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    
    )
   


    args = Parser.parse_args()
    # NumEpochs = Args.NumEpochs
    # BasePath = Args.BasePath
    # DivTrain = float(Args.DivTrain)
    # MiniBatchSize = Args.MiniBatchSize
    # LoadCheckPoint = Args.LoadCheckPoint
    # CheckPointPath = Args.CheckPointPath
    # LogsPath = Args.LogsPath
    # ModelType = Args.ModelType
    print("hi lets begin")
    if args.mode == 'Train':
        Training(args)

    # else:
    #     Test(args)




if __name__ == "__main__":
    main()


