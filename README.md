# sign-language-recognition-translation


## Installation
This code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) v1.0.0 and requires all of its dependencies (`torch==1.6.0`). Additional requirements are [NLTK](https://www.nltk.org/) for NMT evaluation metrics.

The recommended way to install is shown below:


# Create a new virtual environment in Anaconda Navigator

```conda create -n``` _[virtual_env_name]_ ```python=```_[x.x]_ ```anaconda```

```conda activate``` _[virtual_env_name]_


# Clone the repo using Git

```
git clone https://github.com/isaacgn/sign-language-recognition-translation.git
cd sign-language-recognition-translation
```

# Install python dependencies

```
pip install -r requirements.txt
```

# Install OpenNMT-py

```
python setup.py install
```


### Data processing

```
onmt_preprocess -train_src data/phoenix2014T.train.gloss -train_tgt data/phoenix2014T.train.de -valid_src data/phoenix2014T.dev.gloss -valid_tgt data/phoenix2014T.dev.de -save_data data/dgs -lower 
```

### Training
```
python  train.py -data data/dgs -save_model model -keep_checkpoint 1 \
          -layers 2 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
          -encoder_type transformer -decoder_type transformer -position_encoding \
          -max_generator_batches 2 -dropout 0.1 \
          -early_stopping 3 -early_stopping_criteria accuracy ppl \
          -batch_size 2048 -accum_count 3 -batch_type tokens -normalization tokens \
          -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 3000 -learning_rate 0.5 \
          -max_grad_norm 0 -param_init 0  -param_init_glorot \
          -label_smoothing 0.1 -valid_steps 100 -save_checkpoint_steps 100 \
          -world_size 1 
```


### Inference
```python translate.py -model``` _["File path of the model"]_ ```-src``` _["File path of the source data file"]_ ```-output pred.txt -replace_unk -beam_size 4```


### Scoring
```
# BLEU-1,2,3,4
python tools/bleu.py 1 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 2 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 3 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 4 pred.txt data/phoenix2014T.test.de

# ROUGE
python tools/rouge.py pred.txt data/phoenix2014T.test.de

```

