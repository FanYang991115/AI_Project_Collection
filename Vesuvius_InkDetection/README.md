# Vesuvius Ink Detection (Work In Progress)

This repository contains our ongoing work on the Vesuvius Challenge: Ink Detection competition on Kaggle. The goal of this competition is to develop a machine learning model to perform segmentation of ancient ink traces on fossilized layers of volcanic ash.

## Model

We are currently using a simple U-Net architecture as our model for this segmentation task. The U-Net architecture is a popular choice for segmentation tasks due to its efficiency and effectiveness in capturing both high-level and low-level features.

## Data

The dataset provided by the competition is a 3D stack of images, representing different layers of volcanic ash fossils. However, we have decided to focus only on a few middle layers, as these are the layers that contain positive samples (i.e., ink traces). By focusing on these layers, we can concentrate our model's learning on the most relevant data.

## Training

During the training phase, we have experimented with both single-GPU and multi-GPU configurations to optimize model performance and training speed. We continue to fine-tune and explore different training strategies to further improve our model's performance.

## Current Progress

As of now, our team ranks around the top 30% on the Kaggle leaderboard. We are actively working on improving our model and climbing the ranks.

