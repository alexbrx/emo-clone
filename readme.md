## Disentangled Representation Learning and Generative Adversarial Networks for Emotional Voice Cloning


### Motivation
* Given a recorded speech sample we would like to generate new samples havig some qualitative aspects like speaker's voice timbre, prosody, emotion etc. altered.
* Naive application of state-of-the-art GANs for image style transfer doesn't deliver good results because these in general are not well suited to handle sequential data like speech.
* 

### Model Outline
* [SpeechSplit](https://arxiv.org/abs/2004.11284) is an autoencoder neural network whcih decomposes speech into disentangled latent representations corresponding to four main perceptual aspects of speech i.e. pitch, rhythm, lexical content and speaker's voice timbre. 
* The latents can be synthesized back into speech, hence it may be possible to perform style transfer by simply substituting some of the latents to synthesize altered samples.
* Authors of SpeechSplit confirmed that this method works if the latents are substitued between parallel utterances (i.e. same linguistic content).
* 


### Datasets
### Dependencies
See [requirements.txt](requirements.txt)

<!---
## Training WaveGlow
Run the script below
```bash
python waveglow/train.py -c config.json
```
## Training SpeechSplit
Run the script below
```bash
python SpeechSplit/main.py
```
## Training VoiceGAN
Run the script below
```bash
python cvoicegan/main_wgan.py
```
## Training CodeGAN
Run the script below
```bash
python codegan/main_stargan.py
```
## Generate Samples
Run the script below
```bash
python utils/fake_cvoice_samples.py
```
Code mostly or entirely written by me includes
* codegan/*
* cvoicegan/*
* utils/*
* notebooks/*

Code that was in some part rewritten by me
* waveglow/mel2samp.py
* waveglow/inference.py
* waveglow/train.py
* SpeechSplit/make_metadata.py
* SpeechSplit/make_spect_f0.py

The remaining files in waveglow/ and SpeechSplit/ were not changed.

samples.tar.gz contains sample outputs from the model


Original Repos:
https://github.com/auspicious3000/SpeechSplit
https://github.com/NVIDIA/waveglow
https://github.com/yunjey/stargan (codegan/ and cvoicegan/ follow the general structure)
--->
