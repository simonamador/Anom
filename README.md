This is an ongoing project...

<h1 align="center">
  <br>
Anomaly-Detection: Unsupervised learning for detecting structural pathologies in fetal MRI
  <br>
</h1>
  <p align="center">
    <a href="https://github.com/simonamador">Carlos Simon Amador</a> • 
    <a href="https://github.com/VictorSungminYou">Sungmin You</a> •
    <a href="https://github.com/GuillermoTafoya">Guillermo Tafoya</a>
    

## Purpose
Design a framework to localize structural brain anomalies in fetal MRI images through the use of unsupervised learning models.

## Model
Our generation model consists of a variational autoencoder (VAE) composed of 4 convolutional blocks on the encoder and decoder. 
Each convolutional block consists of a convolutional (or transposed convolutional) layer, a leaky ReLU activation, and a batch normalization layer. 
The latent vector produced by the encoder is of size 1x512. The decoder contains a final ReLU activation layer. Three models were trained, one for each view, 
for 2000 epochs with an Adam optimizer, learning rate of 1 x 10^-4, weight decay of 1 x 10^-5. Several loss functions are being tested, including L2, SSIM, and combinations.

Other models being tested include the beta-VAE, and VAE which includes the gestational age as a factor.

![Architecture](/assets/Complete_Framework.png)
