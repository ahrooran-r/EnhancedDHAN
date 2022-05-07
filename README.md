# Enhanced DHAN

### _A modified Dual Hierarchical Network to remove Shadows from Documents with Reflective Surface and Textured Backgrounds such as Sri Lankan National Identity Card (NIC)_

This is a modified DHAN network with alteration to its base architecture based on Ghost Free Shadow Removal by Vinthony (https://github.com/vinthony/ghost-free-shadow-removal)

### Key Limitations with DHAN

#### 1.  Purpose of VGG19 in baseline

- DHAN is designed to tackle natural image shadows. 
- It does a transfer learning using object detection / feature map (VGG19) in its pipeline to remove shadows.
- However, VGG19 is developed with the intent of classification of image into some objects (such as face, hair etc.)
- Our NIC dataset has neither of these features. Thus, creating a feature map to our image will result in wash out of useful textures and letters.

#### 2. Performance of VGG-19

- Since VGG-19 is a CNN with 19 layers deep, it has difficulties working with high resolution images
- However practical photos taken with smartphones can produce high resolution images

## Our Model

![Picture2](https://user-images.githubusercontent.com/46846338/167256407-ddea1a4b-3cfd-43cd-ab6b-75cea37efa00.svg)

![image](https://user-images.githubusercontent.com/46846338/167256427-e2c35e01-9cf8-410a-a9a9-b5ab7bc0cb4c.png)

- Instead of sending source image via VGG-19 and then into DHAN, we directly send source image into DHAN.
- Then to teach the GAN network to omit background textures, we send both source as well as taget images into discriminator and create a perception loss which the generator can use to adjust its next iteration of learning.
- We are in process of publishing our paper with regards to this project

## Summary

![Picture3](https://user-images.githubusercontent.com/46846338/167256455-3047bcb5-3ec0-4178-b210-4dd8a1cb817c.svg)
