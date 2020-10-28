## Disentangled Representation Learning and Generative Adversarial Networks for Emotional Voice Cloning

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
