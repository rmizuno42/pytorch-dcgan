from __future__ import print_function

import os
import numpy as np
import cv2

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from sklearn.utils import shuffle


def load_dataset_from_dir(dir_path):
    print("load data from {}".format(dir_path))
    tmp = []
    for i in os.listdir(dir_path):
        if i.find("jpg") != -1:
            image = cv2.imread(os.path.join(dir_path, i))
            resized = cv2.resize(image, (64, 64))
            tmp.append((resized / 127.5) - 1)
    return tmp


def load_dataset(dir_path_list):
    tmp = []
    for dir_path in dir_path_list:
        tmp += load_dataset_from_dir(dir_path)
    array_data = np.asarray(tmp, dtype=np.float32)
    datasets = np.rollaxis(array_data, 3, 1)
    tmp = np.copy(datasets)
    datasets[:, 0, :, :] = tmp[:, 2, :, :]
    datasets[:, 1, :, :] = tmp[:, 1, :, :]
    datasets[:, 2, :, :] = tmp[:, 0, :, :]
    return datasets


def nparray_to_cuda_variable(x):
    x = torch.from_numpy(x)
    x = x.cuda()
    x = Variable(x)
    return x


def get_predict(output):
    pred = output.data.view(-1)
    predict = (pred * 2.0).floor()
    predict = predict.type_as(torch.LongTensor(1))
    return predict


def get_acc_sum(predict, target):
    correct = predict.eq(target.data.type_as(torch.LongTensor(1))).cpu().sum()
    return correct


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dconv1 = nn.ConvTranspose2d(self.z_dim, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.dconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.dconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.dconv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.dconv5 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = x.view(-1, self.z_dim, 1, 1)
        x = F.relu(self.bn1(self.dconv1(x)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn4(self.dconv4(x)))
        x = F.tanh(self.dconv5(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        # self.fc5 = nn.Linear(512 * 4 * 4, 1)
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2, inplace=True)
        # x = F.sigmoid(self.fc5(x))
        x = F.sigmoid(self.conv5(x))
        return x


class DCGAN(nn.Module):

    def __init__(self, generator, discriminator):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x


def train(dir_path, epochs=100, batch_size=100, z_dim=100, generator_train_times=1, discriminator_train_times=1, output_dir="generated_images"):
    seed = 1
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_x = load_dataset(dir_path)
    nb_train_x = len(train_x)
    nb_batchs = nb_train_x // batch_size

    generator = Generator(z_dim)
    discriminator = Discriminator()
    dcgan = DCGAN(generator, discriminator)
    dcgan.cuda()

    criterion = nn.BCELoss()
    criterion.cuda()

    generator_optimizer = optim.Adam(dcgan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(
        dcgan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    const_data_z = nparray_to_cuda_variable(
        np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32))

    for epoch in range(1, epochs + 1):
        print("----------------------------------------")
        print("------epoch {}---------------------------".format(epoch))
        print("----------------------------------------")
        np.random.shuffle(train_x)
        g_loss_sum = 0
        d_fake_loss_sum = 0
        d_real_loss_sum = 0
        g_acc_sum = 0
        d_acc_sum = 0

        # for batch_index, data_x in enumerate(trainloader):
        for batch_index in range(nb_batchs):
            data_x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            data_x = nparray_to_cuda_variable(data_x)
            data_z = nparray_to_cuda_variable(
                np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32))

            y_zeros = nparray_to_cuda_variable(np.array([0] * batch_size).astype(np.float32))
            y_ones = nparray_to_cuda_variable(np.array([1] * batch_size).astype(np.float32))

            for i in range(generator_train_times):
                generator_optimizer.zero_grad()
                output = dcgan(data_z)
                g_loss = criterion(output, y_zeros)
                g_loss.backward()
                generator_optimizer.step()

            predict = get_predict(output)
            correct = get_acc_sum(predict, y_zeros)
            g_acc = correct / batch_size

            discriminator_optimizer.zero_grad()
            output = dcgan.discriminator(data_x)
            d_real_loss = criterion(output, y_zeros)
            d_real_loss.backward()

            predict = get_predict(output)
            correct = get_acc_sum(predict, y_zeros)
            d_real_acc = correct / batch_size
            discriminator_optimizer.step()

            discriminator_optimizer.zero_grad()
            generated_images = dcgan.generator(data_z)
            output = dcgan.discriminator(generated_images)
            d_fake_loss = criterion(output, y_ones)
            d_fake_loss.backward()
            discriminator_optimizer.step()

            predict = get_predict(output)
            correct = get_acc_sum(predict, y_ones)
            d_fake_acc = correct / batch_size

            d_acc = (d_real_acc + d_fake_acc) / 2

            print("g_loss: {}, g_acc: {}".format(
                g_loss.data[0], g_acc))
            print("d_real_loss: {}, d_fake_loss: {}".format(
                d_real_loss.data[0], d_fake_loss.data[0]))
            print("d_real_acc: {}, d_fake_acc: {}, d_acc: {}".format(d_real_acc,d_fake_acc, d_acc))

            g_loss_sum += g_loss.data[0]
            d_fake_loss_sum += d_fake_loss.data[0]
            d_real_loss_sum += d_real_loss.data[0]
            g_acc_sum += g_acc
            d_acc_sum += d_acc
        print("epoch {} summary".format(epoch))
        print("generator loss: {}, generator acc: {}".format(g_loss_sum/nb_batchs, g_acc_sum / nb_batchs))
        print("discriminator real loss: {}, discriminator fake loss: {}\n discriminator acc: {}".format(
            d_real_loss_sum/nb_batchs, d_fake_loss_sum/nb_batchs, d_acc_sum / nb_batchs))
        if epoch % 10 == 0:
            generated_images = dcgan.generator(const_data_z)
            vutils.save_image(generated_images.data, os.path.join(
                output_dir, "epoch_{}.png".format(epoch)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dirpath', nargs='*')
    parser.add_argument('--gtimes', type=int)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()
    train(args.dirpath, epochs=args.epoch, generator_train_times=args.gtimes)
