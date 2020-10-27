wget -O pretrained.tar.gz https://www.dropbox.com/s/p26daq4uqm1yszs/pretrained.tar.gz?dl=0

tar -xvf pretrained.tar.gz

mkdir -p ./SpeechSplit/assets
mv ./pretrained/iemocap_meta.csv ./SpeechSplit/assets

mkdir -p ./SpeechSplit/run_full/models
mv ./pretrained/2800000-G.ckpt ./SpeechSplit/run_full/models

mkdir -p ./waveglow/checkpoints
mv ./pretrained/waveglow_128000 ./waveglow/checkpoints

mkdir -p ./cvoicegan/experiments/models
mv ./pretrained/1000000-G.ckpt ./cvoicegan/experiments/models

rm -r pretrained
rm pretrained.tar.gz
