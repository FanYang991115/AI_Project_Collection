<!--
 * @Author: FanYang991115 f.yang1@ufl.edu
 * @Date: 2023-05-02 18:43:06
 * @LastEditors: FanYang991115 f.yang1@ufl.edu
 * @LastEditTime: 2023-05-02 18:47:13
 * @FilePath: /projects/ImageColorization/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Image Colorization

This repository contains the implementation of an image colorization project, which aims to automatically colorize grayscale images using deep learning techniques. The model is designed to work with a variety of input images, such as faces and fruits.

## Model

We are using a U-Net and FCN architecture for this colorization task. The U-Net model is a popular choice for image segmentation and colorization tasks due to its efficiency and effectiveness in capturing both high-level and low-level features.

## Data

The dataset for this project consists of images of faces and fruits. The input images are grayscale versions of the original RGB images, while the target images are the full-color RGB versions. Our model's goal is to learn the mapping from grayscale input images to their corresponding full-color target images.

## Workflow

The workflow of this project is as follows:

1. Preprocess the input images by converting them to grayscale.
2. Train the U-Net model on the preprocessed dataset, learning to map grayscale images to their corresponding full-color targets.
3. Use the trained model to automatically colorize new grayscale images.

## Results

The model is capable of producing realistic colorizations for a variety of input images, including faces and fruits. By leveraging the power of deep learning, our model is able to learn the intricate relationships between grayscale and color images, enabling it to generate high-quality colorized outputs.