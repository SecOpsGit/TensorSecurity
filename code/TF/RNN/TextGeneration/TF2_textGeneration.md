#
```
使用RNN生成文本實戰：莎士比亞風格詩句 (tensorflow2.0官方教程翻譯）
https://zhuanlan.zhihu.com/p/69072719

Text generation with an RNN

https://www.tensorflow.org/tutorials/text/text_generation


```

```
本教程演示了如何使用基於字元的 RNN 生成文本[Char-CNN]。

我們將使用 Andrej Karpathy 在 The Unreasonable Effectiveness of Recurrent Neural Networks 一文中提供的莎士比亞作品資料集。
我們根據此資料（“Shakespear”）中的給定字元序列訓練一個模型，讓它預測序列的下一個字元（“e”）。
通過重複調用該模型，可以生成更長的文本序列。

注意：啟用 GPU 加速可提高執行速度。

本教程中包含使用 tf.keras 和 Eager Execution 實現的可運行代碼。
```
```
以下是本教程中的模型訓練了30個週期時的示例輸出，並以字串“Q”開頭：


QUEENE:
I had thought thou hadst a Roman; for the oracle,
Thus by All bids the man against the word,
Which are so weak of care, by old care done;
Your children were in your holy love,
And the precipitation through the bleeding throne.

BISHOP OF ELY:
Marry, and will, my lord, to weep in such a one were prettiest;
Yet now I was adopted heir
Of the world's lamentable day,
To watch the next way with his father with his face?

ESCALUS:
The cause why then we are all resolved more sons.

VOLUMNIA:
O, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, it is no sin it should be dead,
And love and pale as any will to that word.

QUEEN ELIZABETH:
But how long have I heard the soul for this world,
And show his hands of life be proved to stand.

PETRUCHIO:
I say he look'd on, if I must be content
To stay him from the fatal of our country's bliss.
His lordship pluck'd from this sentence then for prey,
And then let us twain, being the moon,
were she such a case as fills m
```

```
雖然有些句子合乎語法規則，但大多數句子都沒有意義。
該模型尚未學習單詞的含義，

但請考慮以下幾點：
該模型是基於字元的模型。在訓練之初，該模型都不知道如何拼寫英語單詞，甚至不知道單詞是一種文本單位。
輸出的文本結構仿照了劇本的結構：文字區塊通常以講話者的名字開頭，並且像資料集中一樣，這些名字全部採用大寫字母。
如下文所示，儘管該模型只使用小批次的文本（每批文本包含 100 個字元）訓練而成，但它仍然能夠生成具有連貫結構的更長文本序列。
```
```

```
