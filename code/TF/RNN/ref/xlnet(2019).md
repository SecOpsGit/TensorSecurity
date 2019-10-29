#
```
XLNet: Generalized Autoregressive Pretraining for Language Understanding
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
(Submitted on 19 Jun 2019)
https://arxiv.org/abs/1906.08237

https://github.com/zihangdai/xlnet
```
```
2019-NLP最強模型: XLNet
WenWei Kang
Jul 8 · 14 min read
在2019年6月中旬Google提出一個NLP模型XLNet，在眾多NLP任務包括RACE, GLUE Benchmark以及許多Text-classification上輾壓眾生，
尤其是在號稱最困難的大型閱讀理解QA任務RACE足足超越BERT 6~9個百分點，其中XLNet模型改善了ELMo, GPT, BERT的缺點，
有ELMo, GPT的AR性質，
又有跟BERT一樣，使用AE性質能夠捕捉bidirectional context的訊息，
最後再把Transformer-XL能夠訓練大型文本的架構拿來用


https://kknews.cc/zh-tw/tech/vr4xxz2.html
https://easyai.tech/blog/nlp-xlnet-bert/

https://zhuanlan.zhihu.com/p/70257427

```
### youtube
```
XLNet: Generalized Autoregressive Pretraining for Language Understanding
https://www.youtube.com/watch?v=H5vpBCLo74U

自然语言理解模型XLNet 杨植麟 01  https://www.youtube.com/watch?v=u275uPAAxN8

自然语言理解模型XLNet 杨植麟 03  https://www.youtube.com/watch?v=zKRkTo9sIgQ

```
### 殘念20191030
```
https://colab.research.google.com/github/graykode/xlnet-Pytorch/blob/master/XLNet.ipynb#scrollTo=BkV6fPyArNbN
```
```
!git clone https://github.com/graykode/xlnet-Pytorch

%cd xlnet-Pytorch

!pip install pytorch_pretrained_bert

!python main.py
```
```
!python main.py --data ./data.txt --tokenizer bert-base-uncased \
   --seq_len 512 --reuse_len 256 --perm_size 256 \
   --bi_data True --mask_alpha 6 --mask_beta 1 \
   --num_predict 85 --mem_len 384 --num_epoch 100
```
```
Traceback (most recent call last):
  File "main.py", line 89, in <module>
    num_predict=args.num_predict)
  File "/content/xlnet-Pytorch/data_utils.py", line 345, in make_permute
    reuse_len)
  File "/content/xlnet-Pytorch/data_utils.py", line 292, in _local_perm
    non_mask_tokens = (~is_masked) & non_func_tokens
RuntimeError: Expected object of scalar type Byte but got scalar type Bool for argument #2 'other' in call to _th_and

```
