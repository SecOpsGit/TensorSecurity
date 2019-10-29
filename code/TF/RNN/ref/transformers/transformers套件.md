# Transformer
```
State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch

🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) 
provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, CTRL...) 
for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 
32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch.

Features
As easy to use as pytorch-transformers
As powerful and concise as Keras
High performance on NLU and NLG tasks
Low barrier to entry for educators and practitioners
State-of-the-art NLP for everyone

Deep learning researchers
Hands-on practitioners
AI/ML/NLP teachers and educators
Lower compute costs, smaller carbon footprint

Researchers can share trained models instead of always retraining
Practitioners can reduce compute time and production costs
10 architectures with over 30 pretrained models, some in more than 100 languages
Choose the right framework for every part of a model's lifetime

Train state-of-the-art models in 3 lines of code
Deep interoperability between TensorFlow 2.0 and PyTorch models
Move a single model between TF2.0/PyTorch frameworks at will
Seamlessly pick the right framework for training, evaluation, production
```
### Write With Transformer

```
Get a modern neural network to auto-complete your thoughts.
This web app, built by the Hugging Face team, is 
the official demo of the 🤗/transformers repository's text generation capabilities.
https://transformer.huggingface.co/
```
### 測試1
```
%tensorflow_version 2.x

!pip install transformers

!git clone https://github.com/huggingface/transformers.git

%cd transformers

%cd examples
```
顯示畫面
```
benchmarks.py	   run_glue.py		   tests_samples
contrib		   run_lm_finetuning.py    utils_multiple_choice.py
distillation	   run_multiple_choice.py  utils_ner.py
README.md	   run_ner.py		   utils_squad_evaluate.py
requirements.txt   run_squad.py		   utils_squad.py
run_bertology.py   run_tf_glue.py
run_generation.py  test_examples.py
```

```
!python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
```
```
2019-10-29 16:55:05.190065: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2019-10-29 16:55:05.224650: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2000129999 Hz
2019-10-29 16:55:05.224928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2599500 executing computations on platform Host. Devices:
2019-10-29 16:55:05.224969: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
10/29/2019 16:55:05 - INFO - transformers.file_utils -   https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json not found in cache or force_download set to True, downloading to /tmp/tmpavz9qktp
100% 1042301/1042301 [00:00<00:00, 2087009.30B/s]
10/29/2019 16:55:06 - INFO - transformers.file_utils -   copying /tmp/tmpavz9qktp to cache at /root/.cache/torch/transformers/f2808208f9bec2320371a9f5f891c184ae0b674ef866b79c58177067d15732dd.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71
10/29/2019 16:55:06 - INFO - transformers.file_utils -   creating metadata file for /root/.cache/torch/transformers/f2808208f9bec2320371a9f5f891c184ae0b674ef866b79c58177067d15732dd.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71
10/29/2019 16:55:06 - INFO - transformers.file_utils -   removing temp file /tmp/tmpavz9qktp
10/29/2019 16:55:07 - INFO - transformers.file_utils -   https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt not found in cache or force_download set to True, downloading to /tmp/tmpl9p0uzvd
100% 456318/456318 [00:00<00:00, 1085845.59B/s]
10/29/2019 16:55:07 - INFO - transformers.file_utils -   copying /tmp/tmpl9p0uzvd to cache at /root/.cache/torch/transformers/d629f792e430b3c76a1291bb2766b0a047e36fae0588f9dbc1ae51decdff691b.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda
10/29/2019 16:55:07 - INFO - transformers.file_utils -   creating metadata file for /root/.cache/torch/transformers/d629f792e430b3c76a1291bb2766b0a047e36fae0588f9dbc1ae51decdff691b.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda
10/29/2019 16:55:07 - INFO - transformers.file_utils -   removing temp file /tmp/tmpl9p0uzvd
10/29/2019 16:55:07 - INFO - transformers.tokenization_utils -   loading file https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json from cache at /root/.cache/torch/transformers/f2808208f9bec2320371a9f5f891c184ae0b674ef866b79c58177067d15732dd.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71
10/29/2019 16:55:07 - INFO - transformers.tokenization_utils -   loading file https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt from cache at /root/.cache/torch/transformers/d629f792e430b3c76a1291bb2766b0a047e36fae0588f9dbc1ae51decdff691b.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda
10/29/2019 16:55:08 - INFO - transformers.file_utils -   https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json not found in cache or force_download set to True, downloading to /tmp/tmpec_w6sfv
100% 176/176 [00:00<00:00, 116324.85B/s]
10/29/2019 16:55:08 - INFO - transformers.file_utils -   copying /tmp/tmpec_w6sfv to cache at /root/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.085d5f6a8e7812ea05ff0e6ed0645ab2e75d80387ad55c1ad9806ee70d272f80
10/29/2019 16:55:08 - INFO - transformers.file_utils -   creating metadata file for /root/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.085d5f6a8e7812ea05ff0e6ed0645ab2e75d80387ad55c1ad9806ee70d272f80
10/29/2019 16:55:08 - INFO - transformers.file_utils -   removing temp file /tmp/tmpec_w6sfv
10/29/2019 16:55:08 - INFO - transformers.configuration_utils -   loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json from cache at /root/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.085d5f6a8e7812ea05ff0e6ed0645ab2e75d80387ad55c1ad9806ee70d272f80
10/29/2019 16:55:08 - INFO - transformers.configuration_utils -   Model config {
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 50257
}

10/29/2019 16:55:09 - INFO - transformers.file_utils -   https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin not found in cache or force_download set to True, downloading to /tmp/tmpia7n84ko
100% 548118077/548118077 [00:17<00:00, 32029148.09B/s]
10/29/2019 16:55:26 - INFO - transformers.file_utils -   copying /tmp/tmpia7n84ko to cache at /root/.cache/torch/transformers/4295d67f022061768f4adc386234dbdb781c814c39662dd1662221c309962c55.778cf36f5c4e5d94c8cd9cefcf2a580c8643570eb327f0d4a1f007fab2acbdf1
10/29/2019 16:55:28 - INFO - transformers.file_utils -   creating metadata file for /root/.cache/torch/transformers/4295d67f022061768f4adc386234dbdb781c814c39662dd1662221c309962c55.778cf36f5c4e5d94c8cd9cefcf2a580c8643570eb327f0d4a1f007fab2acbdf1
10/29/2019 16:55:28 - INFO - transformers.file_utils -   removing temp file /tmp/tmpia7n84ko
10/29/2019 16:55:28 - INFO - transformers.modeling_utils -   loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin from cache at /root/.cache/torch/transformers/4295d67f022061768f4adc386234dbdb781c814c39662dd1662221c309962c55.778cf36f5c4e5d94c8cd9cefcf2a580c8643570eb327f0d4a1f007fab2acbdf1
10/29/2019 16:55:32 - INFO - __main__ -   Namespace(device=device(type='cpu'), length=20, model_name_or_path='gpt2', model_type='gpt2', n_gpu=0, no_cuda=False, padding_text='', prompt='', repetition_penalty=1.0, seed=42, stop_token=None, temperature=1.0, top_k=0, top_p=0.9, xlm_lang='')
Model prompt >>> a
100% 20/20 [00:03<00:00,  6.15it/s]
, and no longer recommends cross-sex relationships. Others have recommended better training for workplace employees. A
Model prompt >>> b
100% 20/20 [00:02<00:00,  6.05it/s]
. Speed = O( r)) Your trade with Pelvis is fast, so my speed would be
Model prompt >>> hello
100% 20/20 [00:02<00:00,  6.22it/s]
" page on the Hartford Courant and its media partner.

"Boston FC's summer campaign
```
