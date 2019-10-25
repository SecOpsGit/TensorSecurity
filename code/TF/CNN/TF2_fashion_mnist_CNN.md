#
```
https://zhuanlan.zhihu.com/p/59506238

https://zhuanlan.zhihu.com/p/59507137
```
```
%tensorflow_version 2.x
```
```
定義三個functions:
def make_model(n_classes)
def load_data()
def train()
```

```
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def make_model(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes)
    ])


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = tf.expand_dims(train_x, -1)
    train_x = (tf.image.convert_image_dtype(train_x, tf.float32) - 0.5) * 2
    train_y = tf.expand_dims(train_y, -1)

    test_x = test_x / 255. * 2 - 1
    test_x = (tf.image.convert_image_dtype(test_x, tf.float32) - 0.5) * 2
    test_y = tf.expand_dims(test_y, -1)

    return (train_x, train_y), (test_x, test_y)


def train():
    # Define the model
    n_classes = 10
    model = make_model(n_classes)

    # Input data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Training parameters
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name="global_step")
    optimizer = tf.optimizers.Adam(1e-3)

    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    accuracy = tf.metrics.Accuracy()
    mean_loss = tf.metrics.Mean(name='loss')

    # Train step function
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        # TODO: apply gradient clipping here
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy.update_state(labels, tf.argmax(logits, -1))
        return loss_value, accuracy.result()

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    train_summary_writer = tf.summary.create_file_writer('./log/train')

    with train_summary_writer.as_default():
        for epoch in range(epochs):
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size

                features, labels = train_x[start_from:to], train_y[start_from:
                                                                   to]

                loss_value, accuracy_value = train_step(features, labels)
                mean_loss.update_state(loss_value)

                if t % 10 == 0:
                    print(
                        f"{step.numpy()}: {loss_value} - accuracy: {accuracy_value}"
                    )
                    save_path = manager.save()
                    print(f"Checkpoint saved: {save_path}")
                    tf.summary.image(
                        'train_set', features, max_outputs=3, step=step.numpy())
                    tf.summary.scalar(
                        'accuracy', accuracy_value, step=step.numpy())
                    tf.summary.scalar(
                        'loss', mean_loss.result(), step=step.numpy())
                    accuracy.reset_states()
                    mean_loss.reset_states()
            print(f"Epoch {epoch} terminated")
            # Measuring accuracy on the whole training set at the end of epoch
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                features, labels = train_x[start_from:to], train_y[start_from:
                                                                   to]
                logits = model(features)
                accuracy.update_state(labels, tf.argmax(logits, -1))
            print(f"Training accuracy: {accuracy.result()}")
            accuracy.reset_states()


if __name__ == "__main__":
    train()
```

執行結果
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 1s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
Initializing from scratch.
Batch size: 32
Number of batches per epoch: 1875
2: 2.3001556396484375 - accuracy: 0.125
Checkpoint saved: ./tf_ckpts/ckpt-1
12: 1.3104352951049805 - accuracy: 0.3656249940395355
Checkpoint saved: ./tf_ckpts/ckpt-2
22: 0.8249340057373047 - accuracy: 0.6781250238418579
Checkpoint saved: ./tf_ckpts/ckpt-3
32: 0.7460381984710693 - accuracy: 0.715624988079071
Checkpoint saved: ./tf_ckpts/ckpt-4
42: 0.574049174785614 - accuracy: 0.715624988079071
Checkpoint saved: ./tf_ckpts/ckpt-5
52: 0.6385637521743774 - accuracy: 0.734375
Checkpoint saved: ./tf_ckpts/ckpt-6
62: 0.6621575355529785 - accuracy: 0.762499988079071
Checkpoint saved: ./tf_ckpts/ckpt-7
72: 0.694776713848114 - accuracy: 0.746874988079071
Checkpoint saved: ./tf_ckpts/ckpt-8
82: 0.6887437105178833 - accuracy: 0.7562500238418579
Checkpoint saved: ./tf_ckpts/ckpt-9
92: 0.6005474328994751 - accuracy: 0.7562500238418579
Checkpoint saved: ./tf_ckpts/ckpt-10
102: 0.6154939532279968 - accuracy: 0.753125011920929
Checkpoint saved: ./tf_ckpts/ckpt-11
112: 0.6492156982421875 - accuracy: 0.768750011920929
Checkpoint saved: ./tf_ckpts/ckpt-12
122: 0.5858597755432129 - accuracy: 0.824999988079071
Checkpoint saved: ./tf_ckpts/ckpt-13
132: 0.4348811209201813 - accuracy: 0.8125
Checkpoint saved: ./tf_ckpts/ckpt-14
142: 0.6374108791351318 - accuracy: 0.8218749761581421
Checkpoint saved: ./tf_ckpts/ckpt-15
152: 0.49940991401672363 - accuracy: 0.840624988079071
Checkpoint saved: ./tf_ckpts/ckpt-16
162: 0.691912055015564 - accuracy: 0.8031250238418579
Checkpoint saved: ./tf_ckpts/ckpt-17
172: 0.6895486116409302 - accuracy: 0.765625
Checkpoint saved: ./tf_ckpts/ckpt-18
182: 0.5095551013946533 - accuracy: 0.815625011920929
Checkpoint saved: ./tf_ckpts/ckpt-19
192: 0.43104881048202515 - accuracy: 0.840624988079071
Checkpoint saved: ./tf_ckpts/ckpt-20
202: 0.3221421241760254 - accuracy: 0.828125
Checkpoint saved: ./tf_ckpts/ckpt-21
212: 0.5001454949378967 - accuracy: 0.8031250238418579
Checkpoint saved: ./tf_ckpts/ckpt-22
222: 0.22166885435581207 - accuracy: 0.840624988079071
Checkpoint saved: ./tf_ckpts/ckpt-23
232: 0.28129637241363525 - accuracy: 0.828125
Checkpoint saved: ./tf_ckpts/ckpt-24
242: 0.5006104111671448 - accuracy: 0.8187500238418579
Checkpoint saved: ./tf_ckpts/ckpt-25
252: 0.7336670160293579 - accuracy: 0.8218749761581421
Checkpoint saved: ./tf_ckpts/ckpt-26
262: 0.761843204498291 - accuracy: 0.846875011920929
Checkpoint saved: ./tf_ckpts/ckpt-27
272: 0.5833059549331665 - accuracy: 0.815625011920929
Checkpoint saved: ./tf_ckpts/ckpt-28
282: 0.5406854152679443 - accuracy: 0.815625011920929
Checkpoint saved: ./tf_ckpts/ckpt-29
292: 0.283017098903656 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-30
302: 0.5746729373931885 - accuracy: 0.828125
Checkpoint saved: ./tf_ckpts/ckpt-31
312: 0.26203519105911255 - accuracy: 0.8125
Checkpoint saved: ./tf_ckpts/ckpt-32
322: 0.6357959508895874 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-33
332: 0.21897894144058228 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-34
342: 0.3359435200691223 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-35
352: 0.2241969108581543 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-36
362: 0.6921106576919556 - accuracy: 0.8500000238418579
Checkpoint saved: ./tf_ckpts/ckpt-37
372: 0.9588035345077515 - accuracy: 0.8125
Checkpoint saved: ./tf_ckpts/ckpt-38
382: 0.41361045837402344 - accuracy: 0.8125
Checkpoint saved: ./tf_ckpts/ckpt-39
392: 0.37433499097824097 - accuracy: 0.84375
Checkpoint saved: ./tf_ckpts/ckpt-40
402: 0.2997359335422516 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-41
412: 0.7009783983230591 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-42
422: 0.38282716274261475 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-43
432: 0.390290230512619 - accuracy: 0.8374999761581421
Checkpoint saved: ./tf_ckpts/ckpt-44
442: 0.27816683053970337 - accuracy: 0.824999988079071
Checkpoint saved: ./tf_ckpts/ckpt-45
452: 0.2955979108810425 - accuracy: 0.856249988079071
Checkpoint saved: ./tf_ckpts/ckpt-46
462: 0.4395046830177307 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-47
472: 0.4360043704509735 - accuracy: 0.84375
Checkpoint saved: ./tf_ckpts/ckpt-48
482: 0.23486822843551636 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-49
492: 0.20234660804271698 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-50
502: 0.5645604133605957 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-51
512: 0.3262104392051697 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-52
522: 0.3703077733516693 - accuracy: 0.846875011920929
Checkpoint saved: ./tf_ckpts/ckpt-53
532: 0.2656984329223633 - accuracy: 0.8500000238418579
Checkpoint saved: ./tf_ckpts/ckpt-54
542: 0.41569799184799194 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-55
552: 0.4154290556907654 - accuracy: 0.846875011920929
Checkpoint saved: ./tf_ckpts/ckpt-56
562: 0.48112744092941284 - accuracy: 0.831250011920929
Checkpoint saved: ./tf_ckpts/ckpt-57
572: 0.5743396282196045 - accuracy: 0.815625011920929
Checkpoint saved: ./tf_ckpts/ckpt-58
582: 0.3789936304092407 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-59
592: 0.31837576627731323 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-60
602: 0.26579684019088745 - accuracy: 0.84375
Checkpoint saved: ./tf_ckpts/ckpt-61
612: 0.3225864768028259 - accuracy: 0.856249988079071
Checkpoint saved: ./tf_ckpts/ckpt-62
622: 0.7948295474052429 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-63
632: 0.5528569221496582 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-64
642: 0.38758277893066406 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-65
652: 0.45220157504081726 - accuracy: 0.8125
Checkpoint saved: ./tf_ckpts/ckpt-66
662: 0.23718248307704926 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-67
672: 0.5347179174423218 - accuracy: 0.84375
Checkpoint saved: ./tf_ckpts/ckpt-68
682: 0.23985008895397186 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-69
692: 0.3896949887275696 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-70
702: 0.24305248260498047 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-71
712: 0.19747436046600342 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-72
722: 0.6792322397232056 - accuracy: 0.8500000238418579
Checkpoint saved: ./tf_ckpts/ckpt-73
732: 0.36917927861213684 - accuracy: 0.840624988079071
Checkpoint saved: ./tf_ckpts/ckpt-74
742: 0.42141902446746826 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-75
752: 0.5061593651771545 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-76
762: 0.5527761578559875 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-77
772: 0.27342915534973145 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-78
782: 0.5182101130485535 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-79
792: 0.09202457964420319 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-80
802: 0.452543169260025 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-81
812: 0.31123995780944824 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-82
822: 0.5961300730705261 - accuracy: 0.828125
Checkpoint saved: ./tf_ckpts/ckpt-83
832: 0.2801534831523895 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-84
842: 0.23018741607666016 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-85
852: 0.33699023723602295 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-86
862: 0.33906009793281555 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-87
872: 0.39969348907470703 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-88
882: 0.46547120809555054 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-89
892: 0.3789883255958557 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-90
902: 0.37187957763671875 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-91
912: 0.3046432137489319 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-92
922: 0.40661758184432983 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-93
932: 0.4221400022506714 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-94
942: 0.5006144046783447 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-95
952: 0.411704421043396 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-96
962: 0.3256233036518097 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-97
972: 0.2544291019439697 - accuracy: 0.8500000238418579
Checkpoint saved: ./tf_ckpts/ckpt-98
982: 0.19225077331066132 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-99
992: 0.5180766582489014 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-100
1002: 0.5530182123184204 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-101
1012: 0.24482491612434387 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-102
1022: 0.2980806529521942 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-103
1032: 0.19747507572174072 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-104
1042: 0.2536852955818176 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-105
1052: 0.18007662892341614 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-106
1062: 0.26832741498947144 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-107
1072: 0.2380291223526001 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-108
1082: 0.22743704915046692 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-109
1092: 0.4858749508857727 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-110
1102: 0.27618953585624695 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-111
1112: 0.21653962135314941 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-112
1122: 0.463468998670578 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-113
1132: 0.31047648191452026 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-114
1142: 0.5258488059043884 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-115
1152: 0.25674575567245483 - accuracy: 0.8531249761581421
Checkpoint saved: ./tf_ckpts/ckpt-116
1162: 0.3731677532196045 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-117
1172: 0.29089489579200745 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-118
1182: 0.12662646174430847 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-119
1192: 0.4840608239173889 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-120
1202: 0.27671754360198975 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-121
1212: 0.3646266460418701 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-122
1222: 0.24286247789859772 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-123
1232: 0.2564208507537842 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-124
1242: 0.2441474348306656 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-125
1252: 0.3344643712043762 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-126
1262: 0.2945673167705536 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-127
1272: 0.1394164115190506 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-128
1282: 0.24619583785533905 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-129
1292: 0.34782445430755615 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-130
1302: 0.4115282893180847 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-131
1312: 0.31132519245147705 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-132
1322: 0.50443434715271 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-133
1332: 0.2688390910625458 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-134
1342: 0.4162069261074066 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-135
1352: 0.32000458240509033 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-136
1362: 0.16186948120594025 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-137
1372: 0.3710780143737793 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-138
1382: 0.2836344242095947 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-139
1392: 0.304223895072937 - accuracy: 0.8687499761581421
Checkpoint saved: ./tf_ckpts/ckpt-140
1402: 0.5701140761375427 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-141
1412: 0.31730082631111145 - accuracy: 0.846875011920929
Checkpoint saved: ./tf_ckpts/ckpt-142
1422: 0.2566468119621277 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-143
1432: 0.13079093396663666 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-144
1442: 0.40467119216918945 - accuracy: 0.856249988079071
Checkpoint saved: ./tf_ckpts/ckpt-145
1452: 0.18104375898838043 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-146
1462: 0.2278362661600113 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-147
1472: 0.14690428972244263 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-148
1482: 0.24624760448932648 - accuracy: 0.84375
Checkpoint saved: ./tf_ckpts/ckpt-149
1492: 0.16018542647361755 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-150
1502: 0.32070785760879517 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-151
1512: 0.2577560245990753 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-152
1522: 0.37150993943214417 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-153
1532: 0.25816646218299866 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-154
1542: 0.39205893874168396 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-155
1552: 0.3083553910255432 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-156
1562: 0.19607426226139069 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-157
1572: 0.24348920583724976 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-158
1582: 0.19457873702049255 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-159
1592: 0.3209468126296997 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-160
1602: 0.1573958396911621 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-161
1612: 0.2617819011211395 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-162
1622: 0.34052640199661255 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-163
1632: 0.2985377907752991 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-164
1642: 0.5935094952583313 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-165
1652: 0.17851051688194275 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-166
1662: 0.26879051327705383 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-167
1672: 0.41804373264312744 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-168
1682: 0.27976417541503906 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-169
1692: 0.5578556060791016 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-170
1702: 0.3487583100795746 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-171
1712: 0.18180859088897705 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-172
1722: 0.24354246258735657 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-173
1732: 0.21384240686893463 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-174
1742: 0.36352548003196716 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-175
1752: 0.30333060026168823 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-176
1762: 0.4450768232345581 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-177
1772: 0.08950181305408478 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-178
1782: 0.47303637862205505 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-179
1792: 0.21574322879314423 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-180
1802: 0.29269465804100037 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-181
1812: 0.24170976877212524 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-182
1822: 0.2375270426273346 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-183
1832: 0.2576668858528137 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-184
1842: 0.29638031125068665 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-185
1852: 0.20858991146087646 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-186
1862: 0.3116954267024994 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-187
1872: 0.20574699342250824 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-188
Epoch 0 terminated
Training accuracy: 0.9021254777908325
1877: 0.21552449464797974 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-189
1887: 0.37510865926742554 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-190
1897: 0.215290367603302 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-191
1907: 0.41067424416542053 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-192
1917: 0.2655037045478821 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-193
1927: 0.24829024076461792 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-194
1937: 0.2348749041557312 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-195
1947: 0.36647510528564453 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-196
1957: 0.24137046933174133 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-197
1967: 0.3156047463417053 - accuracy: 0.8656250238418579
Checkpoint saved: ./tf_ckpts/ckpt-198
1977: 0.27182769775390625 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-199
1987: 0.29384952783584595 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-200
1997: 0.44671109318733215 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-201
2007: 0.2209126502275467 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-202
2017: 0.3076593577861786 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-203
2027: 0.2125042974948883 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-204
2037: 0.3136732280254364 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-205
2047: 0.4109288454055786 - accuracy: 0.84375
Checkpoint saved: ./tf_ckpts/ckpt-206
2057: 0.24945758283138275 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-207
2067: 0.23466143012046814 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-208
2077: 0.1812906563282013 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-209
2087: 0.19564752280712128 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-210
2097: 0.12611821293830872 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-211
2107: 0.2925831377506256 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-212
2117: 0.29968780279159546 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-213
2127: 0.4692841172218323 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-214
2137: 0.48411017656326294 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-215
2147: 0.3216830790042877 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-216
2157: 0.2299160659313202 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-217
2167: 0.22313790023326874 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-218
2177: 0.32239729166030884 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-219
2187: 0.07368552684783936 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-220
2197: 0.3734295964241028 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-221
2207: 0.14830324053764343 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-222
2217: 0.17395871877670288 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-223
2227: 0.12342394888401031 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-224
2237: 0.24202553927898407 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-225
2247: 0.4607144594192505 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-226
2257: 0.32541191577911377 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-227
2267: 0.23428481817245483 - accuracy: 0.859375
Checkpoint saved: ./tf_ckpts/ckpt-228
2277: 0.1443796157836914 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-229
2287: 0.5922117233276367 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-230
2297: 0.23964367806911469 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-231
2307: 0.2041463851928711 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-232
2317: 0.2024964839220047 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-233
2327: 0.3289395570755005 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-234
2337: 0.26114964485168457 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-235
2347: 0.2884443998336792 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-236
2357: 0.19152042269706726 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-237
2367: 0.07706452906131744 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-238
2377: 0.4035727381706238 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-239
2387: 0.35498058795928955 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-240
2397: 0.25629717111587524 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-241
2407: 0.3234439492225647 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-242
2417: 0.3026885390281677 - accuracy: 0.871874988079071
Checkpoint saved: ./tf_ckpts/ckpt-243
2427: 0.26846981048583984 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-244
2437: 0.26394861936569214 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-245
2447: 0.36493924260139465 - accuracy: 0.862500011920929
Checkpoint saved: ./tf_ckpts/ckpt-246
2457: 0.28497523069381714 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-247
2467: 0.2381865382194519 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-248
2477: 0.20836767554283142 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-249
2487: 0.27733832597732544 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-250
2497: 0.49011847376823425 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-251
2507: 0.3229960799217224 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-252
2517: 0.42610806226730347 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-253
2527: 0.3376227021217346 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-254
2537: 0.16220535337924957 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-255
2547: 0.316275417804718 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-256
2557: 0.1989850550889969 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-257
2567: 0.30199283361434937 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-258
2577: 0.2227560430765152 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-259
2587: 0.18949155509471893 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-260
2597: 0.36222293972969055 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-261
2607: 0.21831047534942627 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-262
2617: 0.33433598279953003 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-263
2627: 0.3088764250278473 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-264
2637: 0.3222944736480713 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-265
2647: 0.16072185337543488 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-266
2657: 0.36218297481536865 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-267
2667: 0.11428810656070709 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-268
2677: 0.39267030358314514 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-269
2687: 0.2248699963092804 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-270
2697: 0.391630619764328 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-271
2707: 0.13486143946647644 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-272
2717: 0.17267060279846191 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-273
2727: 0.2112383395433426 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-274
2737: 0.16243192553520203 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-275
2747: 0.28577446937561035 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-276
2757: 0.28636714816093445 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-277
2767: 0.20832018554210663 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-278
2777: 0.22539767622947693 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-279
2787: 0.20021559298038483 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-280
2797: 0.20979173481464386 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-281
2807: 0.20462749898433685 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-282
2817: 0.5319666862487793 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-283
2827: 0.3336390554904938 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-284
2837: 0.2772354185581207 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-285
2847: 0.19376471638679504 - accuracy: 0.878125011920929
Checkpoint saved: ./tf_ckpts/ckpt-286
2857: 0.15442995727062225 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-287
2867: 0.44017523527145386 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-288
2877: 0.42654359340667725 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-289
2887: 0.13758358359336853 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-290
2897: 0.2344152331352234 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-291
2907: 0.14737585186958313 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-292
2917: 0.14059282839298248 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-293
2927: 0.08793851733207703 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-294
2937: 0.14959654211997986 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-295
2947: 0.17608653008937836 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-296
2957: 0.18204909563064575 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-297
2967: 0.38228484988212585 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-298
2977: 0.16980424523353577 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-299
2987: 0.17024938762187958 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-300
2997: 0.43614232540130615 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-301
3007: 0.2750704288482666 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-302
3017: 0.41197192668914795 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-303
3027: 0.21074098348617554 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-304
3037: 0.2817403972148895 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-305
3047: 0.1853485107421875 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-306
3057: 0.08160579949617386 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-307
3067: 0.2734931409358978 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-308
3077: 0.26381030678749084 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-309
3087: 0.3029416799545288 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-310
3097: 0.20339694619178772 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-311
3107: 0.248487651348114 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-312
3117: 0.14984412491321564 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-313
3127: 0.35924988985061646 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-314
3137: 0.21008983254432678 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-315
3147: 0.08644097298383713 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-316
3157: 0.20693936944007874 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-317
3167: 0.26859164237976074 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-318
3177: 0.3143240511417389 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-319
3187: 0.24927061796188354 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-320
3197: 0.339768648147583 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-321
3207: 0.17225942015647888 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-322
3217: 0.36985838413238525 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-323
3227: 0.07082588225603104 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-324
3237: 0.10629156976938248 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-325
3247: 0.3958823084831238 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-326
3257: 0.20949673652648926 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-327
3267: 0.19773465394973755 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-328
3277: 0.4955364763736725 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-329
3287: 0.26980218291282654 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-330
3297: 0.21234285831451416 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-331
3307: 0.09082003682851791 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-332
3317: 0.30443304777145386 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-333
3327: 0.13365203142166138 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-334
3337: 0.22558113932609558 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-335
3347: 0.1713678538799286 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-336
3357: 0.2554684281349182 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-337
3367: 0.1313668191432953 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-338
3377: 0.3080550730228424 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-339
3387: 0.19896310567855835 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-340
3397: 0.24602928757667542 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-341
3407: 0.1851923167705536 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-342
3417: 0.2575356066226959 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-343
3427: 0.18703597784042358 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-344
3437: 0.17851103842258453 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-345
3447: 0.1346338987350464 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-346
3457: 0.07977023720741272 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-347
3467: 0.22159509360790253 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-348
3477: 0.10594918578863144 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-349
3487: 0.21468593180179596 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-350
3497: 0.34740692377090454 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-351
3507: 0.1898004412651062 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-352
3517: 0.5658817291259766 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-353
3527: 0.1230681985616684 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-354
3537: 0.2821881175041199 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-355
3547: 0.31761324405670166 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-356
3557: 0.15484397113323212 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-357
3567: 0.2987784445285797 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-358
3577: 0.21793654561042786 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-359
3587: 0.2484789341688156 - accuracy: 0.8843749761581421
Checkpoint saved: ./tf_ckpts/ckpt-360
3597: 0.21391785144805908 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-361
3607: 0.1679186075925827 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-362
3617: 0.2916429042816162 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-363
3627: 0.24453023076057434 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-364
3637: 0.3518803119659424 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-365
3647: 0.09329144656658173 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-366
3657: 0.38636016845703125 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-367
3667: 0.15474730730056763 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-368
3677: 0.21302926540374756 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-369
3687: 0.15857887268066406 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-370
3697: 0.23131048679351807 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-371
3707: 0.1673174500465393 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-372
3717: 0.23505499958992004 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-373
3727: 0.20215290784835815 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-374
3737: 0.1870005577802658 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-375
3747: 0.15843071043491364 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-376
Epoch 1 terminated
Training accuracy: 0.9185404181480408
3752: 0.12493444979190826 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-377
3762: 0.2612321078777313 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-378
3772: 0.17037329077720642 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-379
3782: 0.28718361258506775 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-380
3792: 0.2061421275138855 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-381
3802: 0.25670915842056274 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-382
3812: 0.20210124552249908 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-383
3822: 0.2316199392080307 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-384
3832: 0.18870893120765686 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-385
3842: 0.25857824087142944 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-386
3852: 0.27621373534202576 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-387
3862: 0.17705324292182922 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-388
3872: 0.35146504640579224 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-389
3882: 0.15816332399845123 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-390
3892: 0.27350127696990967 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-391
3902: 0.1074257418513298 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-392
3912: 0.2940102815628052 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-393
3922: 0.26973697543144226 - accuracy: 0.875
Checkpoint saved: ./tf_ckpts/ckpt-394
3932: 0.173836350440979 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-395
3942: 0.2400888055562973 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-396
3952: 0.07712723314762115 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-397
3962: 0.15802425146102905 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-398
3972: 0.146607905626297 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-399
3982: 0.2618432343006134 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-400
3992: 0.16425727307796478 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-401
4002: 0.41810131072998047 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-402
4012: 0.30752137303352356 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-403
4022: 0.23316793143749237 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-404
4032: 0.2522921562194824 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-405
4042: 0.19106782972812653 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-406
4052: 0.2037261426448822 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-407
4062: 0.06566646695137024 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-408
4072: 0.2666053771972656 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-409
4082: 0.07462746649980545 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-410
4092: 0.13887014985084534 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-411
4102: 0.09833047538995743 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-412
4112: 0.16049744188785553 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-413
4122: 0.2980591356754303 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-414
4132: 0.32846158742904663 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-415
4142: 0.1391465663909912 - accuracy: 0.893750011920929
Checkpoint saved: ./tf_ckpts/ckpt-416
4152: 0.1079796701669693 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-417
4162: 0.5133572816848755 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-418
4172: 0.17400816082954407 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-419
4182: 0.1427994817495346 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-420
4192: 0.148377925157547 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-421
4202: 0.29202306270599365 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-422
4212: 0.2034718096256256 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-423
4222: 0.22108915448188782 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-424
4232: 0.11414751410484314 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-425
4242: 0.0800732970237732 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-426
4252: 0.3582790493965149 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-427
4262: 0.23392876982688904 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-428
4272: 0.18696646392345428 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-429
4282: 0.35134050250053406 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-430
4292: 0.2929801940917969 - accuracy: 0.890625
Checkpoint saved: ./tf_ckpts/ckpt-431
4302: 0.16629241406917572 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-432
4312: 0.16286572813987732 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-433
4322: 0.2710212171077728 - accuracy: 0.8812500238418579
Checkpoint saved: ./tf_ckpts/ckpt-434
4332: 0.20241771638393402 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-435
4342: 0.19424830377101898 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-436
4352: 0.23328757286071777 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-437
4362: 0.19868232309818268 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-438
4372: 0.37510353326797485 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-439
4382: 0.13044843077659607 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-440
4392: 0.5085427165031433 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-441
4402: 0.301193505525589 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-442
4412: 0.09650377929210663 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-443
4422: 0.28212523460388184 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-444
4432: 0.22829128801822662 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-445
4442: 0.2638021409511566 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-446
4452: 0.20126815140247345 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-447
4462: 0.17095309495925903 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-448
4472: 0.3266083598136902 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-449
4482: 0.15419356524944305 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-450
4492: 0.2575949430465698 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-451
4502: 0.2416321188211441 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-452
4512: 0.24351651966571808 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-453
4522: 0.1426289826631546 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-454
4532: 0.28071415424346924 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-455
4542: 0.037275925278663635 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-456
4552: 0.3530791997909546 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-457
4562: 0.28599807620048523 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-458
4572: 0.32080531120300293 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-459
4582: 0.08467209339141846 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-460
4592: 0.09923213720321655 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-461
4602: 0.17015261948108673 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-462
4612: 0.12874919176101685 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-463
4622: 0.21959467232227325 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-464
4632: 0.19953516125679016 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-465
4642: 0.22128283977508545 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-466
4652: 0.1559840738773346 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-467
4662: 0.20093154907226562 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-468
4672: 0.19995279610157013 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-469
4682: 0.14679236710071564 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-470
4692: 0.48354238271713257 - accuracy: 0.8968750238418579
Checkpoint saved: ./tf_ckpts/ckpt-471
4702: 0.2429656982421875 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-472
4712: 0.3051503896713257 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-473
4722: 0.12536996603012085 - accuracy: 0.887499988079071
Checkpoint saved: ./tf_ckpts/ckpt-474
4732: 0.10945673286914825 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-475
4742: 0.3172767758369446 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-476
4752: 0.40309053659439087 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-477
4762: 0.09822177141904831 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-478
4772: 0.15490761399269104 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-479
4782: 0.11060862243175507 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-480
4792: 0.1296859085559845 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-481
4802: 0.07712201029062271 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-482
4812: 0.09267710149288177 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-483
4822: 0.13406354188919067 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-484
4832: 0.13482005894184113 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-485
4842: 0.3716665208339691 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-486
4852: 0.0901017040014267 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-487
4862: 0.1599973738193512 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-488
4872: 0.3011201322078705 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-489
4882: 0.23476120829582214 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-490
4892: 0.40600916743278503 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-491
4902: 0.22928954660892487 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-492
4912: 0.2620600759983063 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-493
4922: 0.20806553959846497 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-494
4932: 0.07166148722171783 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-495
4942: 0.11504250019788742 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-496
4952: 0.2604644298553467 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-497
4962: 0.22143498063087463 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-498
4972: 0.2381787747144699 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-499
4982: 0.19397611916065216 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-500
4992: 0.07767704874277115 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-501
5002: 0.23909364640712738 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-502
5012: 0.18529024720191956 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-503
5022: 0.04115617275238037 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-504
5032: 0.14141149818897247 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-505
5042: 0.1913444697856903 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-506
5052: 0.16389250755310059 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-507
5062: 0.15602195262908936 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-508
5072: 0.29826539754867554 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-509
5082: 0.1381540298461914 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-510
5092: 0.2544417381286621 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-511
5102: 0.03026604652404785 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-512
5112: 0.10808829963207245 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-513
5122: 0.37536442279815674 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-514
5132: 0.16854658722877502 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-515
5142: 0.17531540989875793 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-516
5152: 0.35222798585891724 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-517
5162: 0.19008153676986694 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-518
5172: 0.12750351428985596 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-519
5182: 0.06976122409105301 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-520
5192: 0.1658419370651245 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-521
5202: 0.1767134666442871 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-522
5212: 0.0633588582277298 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-523
5222: 0.1530882567167282 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-524
5232: 0.2748561203479767 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-525
5242: 0.13899755477905273 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-526
5252: 0.317806601524353 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-527
5262: 0.19950991868972778 - accuracy: 0.8999999761581421
Checkpoint saved: ./tf_ckpts/ckpt-528
5272: 0.23503175377845764 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-529
5282: 0.07857939600944519 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-530
5292: 0.22261497378349304 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-531
5302: 0.12963049113750458 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-532
5312: 0.14863497018814087 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-533
5322: 0.12000251561403275 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-534
5332: 0.06680522859096527 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-535
5342: 0.25689882040023804 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-536
5352: 0.07461106032133102 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-537
5362: 0.1979590207338333 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-538
5372: 0.33583128452301025 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-539
5382: 0.1709643304347992 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-540
5392: 0.35501986742019653 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-541
5402: 0.10679954290390015 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-542
5412: 0.18775874376296997 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-543
5422: 0.2276940494775772 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-544
5432: 0.12126430869102478 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-545
5442: 0.1653236746788025 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-546
5452: 0.13766607642173767 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-547
5462: 0.2045978456735611 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-548
5472: 0.1701221913099289 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-549
5482: 0.12133745849132538 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-550
5492: 0.25360602140426636 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-551
5502: 0.15983745455741882 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-552
5512: 0.2632363736629486 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-553
5522: 0.06434138864278793 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-554
5532: 0.3367615342140198 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-555
5542: 0.08632736653089523 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-556
5552: 0.13600769639015198 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-557
5562: 0.11846058815717697 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-558
5572: 0.22311300039291382 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-559
5582: 0.12361100316047668 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-560
5592: 0.21466276049613953 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-561
5602: 0.16108424961566925 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-562
5612: 0.16323252022266388 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-563
5622: 0.1280093938112259 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-564
Epoch 2 terminated
Training accuracy: 0.9264901280403137
5627: 0.07962767034769058 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-565
5637: 0.11317305266857147 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-566
5647: 0.13738298416137695 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-567
5657: 0.28718966245651245 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-568
5667: 0.12695561349391937 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-569
5677: 0.2561322748661041 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-570
5687: 0.2015112042427063 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-571
5697: 0.18433120846748352 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-572
5707: 0.17209041118621826 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-573
5717: 0.1918499767780304 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-574
5727: 0.15992391109466553 - accuracy: 0.903124988079071
Checkpoint saved: ./tf_ckpts/ckpt-575
5737: 0.12631602585315704 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-576
5747: 0.27716243267059326 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-577
5757: 0.10697740316390991 - accuracy: 0.909375011920929
Checkpoint saved: ./tf_ckpts/ckpt-578
5767: 0.21729399263858795 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-579
5777: 0.1671850085258484 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-580
5787: 0.19921153783798218 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-581
5797: 0.10736812651157379 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-582
5807: 0.08938479423522949 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-583
5817: 0.23842409253120422 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-584
5827: 0.06556938588619232 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-585
5837: 0.07367569208145142 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-586
5847: 0.13898426294326782 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-587
5857: 0.2033172994852066 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-588
5867: 0.08586180955171585 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-589
5877: 0.2801814377307892 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-590
5887: 0.171833336353302 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-591
5897: 0.14455094933509827 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-592
5907: 0.18148131668567657 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-593
5917: 0.10967942327260971 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-594
5927: 0.12097421288490295 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-595
5937: 0.06213528662919998 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-596
5947: 0.22803890705108643 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-597
5957: 0.047650061547756195 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-598
5967: 0.07584857195615768 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-599
5977: 0.09582389891147614 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-600
5987: 0.13211072981357574 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-601
5997: 0.1626555174589157 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-602
6007: 0.11631061881780624 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-603
6017: 0.18913356959819794 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-604
6027: 0.13266624510288239 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-605
6037: 0.45158299803733826 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-606
6047: 0.13551804423332214 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-607
6057: 0.09028944373130798 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-608
6067: 0.1444370001554489 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-609
6077: 0.3432796597480774 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-610
6087: 0.12252883613109589 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-611
6097: 0.18221020698547363 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-612
6107: 0.0864216759800911 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-613
6117: 0.06960578262805939 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-614
6127: 0.2555495798587799 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-615
6137: 0.17729774117469788 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-616
6147: 0.148408442735672 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-617
6157: 0.34988611936569214 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-618
6167: 0.27616995573043823 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-619
6177: 0.12589776515960693 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-620
6187: 0.14392229914665222 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-621
6197: 0.2731265127658844 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-622
6207: 0.19036641716957092 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-623
6217: 0.20855969190597534 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-624
6227: 0.15666809678077698 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-625
6237: 0.14995819330215454 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-626
6247: 0.282224178314209 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-627
6257: 0.06486260890960693 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-628
6267: 0.5656974911689758 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-629
6277: 0.2841701805591583 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-630
6287: 0.06274893134832382 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-631
6297: 0.11526557058095932 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-632
6307: 0.13830773532390594 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-633
6317: 0.15779496729373932 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-634
6327: 0.1990334838628769 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-635
6337: 0.1391875147819519 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-636
6347: 0.276050865650177 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-637
6357: 0.10502933710813522 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-638
6367: 0.1245146095752716 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-639
6377: 0.27557438611984253 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-640
6387: 0.16725775599479675 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-641
6397: 0.11506733298301697 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-642
6407: 0.24965554475784302 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-643
6417: 0.02393699623644352 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-644
6427: 0.19403813779354095 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-645
6437: 0.2577317953109741 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-646
6447: 0.21939747035503387 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-647
6457: 0.03735636547207832 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-648
6467: 0.06724482774734497 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-649
6477: 0.14418837428092957 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-650
6487: 0.0751798003911972 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-651
6497: 0.19908374547958374 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-652
6507: 0.16939379274845123 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-653
6517: 0.1442035734653473 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-654
6527: 0.09717397391796112 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-655
6537: 0.16475367546081543 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-656
6547: 0.1355525255203247 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-657
6557: 0.10765130817890167 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-658
6567: 0.34382694959640503 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-659
6577: 0.1899014413356781 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-660
6587: 0.2916822135448456 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-661
6597: 0.06072097271680832 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-662
6607: 0.05751580372452736 - accuracy: 0.9125000238418579
Checkpoint saved: ./tf_ckpts/ckpt-663
6617: 0.36957427859306335 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-664
6627: 0.4258052110671997 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-665
6637: 0.053964629769325256 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-666
6647: 0.0619039386510849 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-667
6657: 0.08341168612241745 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-668
6667: 0.12613384425640106 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-669
6677: 0.053650595247745514 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-670
6687: 0.059703364968299866 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-671
6697: 0.11052168160676956 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-672
6707: 0.11230239272117615 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-673
6717: 0.31106624007225037 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-674
6727: 0.06619725376367569 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-675
6737: 0.1671338677406311 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-676
6747: 0.2224462330341339 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-677
6757: 0.2357477843761444 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-678
6767: 0.33359402418136597 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-679
6777: 0.22767074406147003 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-680
6787: 0.20561790466308594 - accuracy: 0.90625
Checkpoint saved: ./tf_ckpts/ckpt-681
6797: 0.11155112832784653 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-682
6807: 0.07117122411727905 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-683
6817: 0.07251795381307602 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-684
6827: 0.1952298879623413 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-685
6837: 0.19453410804271698 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-686
6847: 0.2180696725845337 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-687
6857: 0.12559902667999268 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-688
6867: 0.05648413300514221 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-689
6877: 0.08866185694932938 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-690
6887: 0.08675556629896164 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-691
6897: 0.029620517045259476 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-692
6907: 0.14120230078697205 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-693
6917: 0.14029230177402496 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-694
6927: 0.12410289794206619 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-695
6937: 0.14226028323173523 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-696
6947: 0.2792646288871765 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-697
6957: 0.10744210332632065 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-698
6967: 0.2212449312210083 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-699
6977: 0.023049429059028625 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-700
6987: 0.09407813847064972 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-701
6997: 0.25009799003601074 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-702
7007: 0.11492297053337097 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-703
7017: 0.17623278498649597 - accuracy: 0.918749988079071
Checkpoint saved: ./tf_ckpts/ckpt-704
7027: 0.19045719504356384 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-705
7037: 0.18311350047588348 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-706
7047: 0.07547911256551743 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-707
7057: 0.0893770083785057 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-708
7067: 0.1115415021777153 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-709
7077: 0.05872379243373871 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-710
7087: 0.047831013798713684 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-711
7097: 0.1541588455438614 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-712
7107: 0.3270304203033447 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-713
7117: 0.08992619812488556 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-714
7127: 0.3576778769493103 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-715
7137: 0.16575993597507477 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-716
7147: 0.17927998304367065 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-717
7157: 0.11217165738344193 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-718
7167: 0.2188122719526291 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-719
7177: 0.09713594615459442 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-720
7187: 0.11491040885448456 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-721
7197: 0.04223781079053879 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-722
7207: 0.06572198122739792 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-723
7217: 0.25763314962387085 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-724
7227: 0.04225223511457443 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-725
7237: 0.1869262158870697 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-726
7247: 0.2866225838661194 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-727
7257: 0.12597210705280304 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-728
7267: 0.24365906417369843 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-729
7277: 0.11338608711957932 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-730
7287: 0.14052489399909973 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-731
7297: 0.1176133006811142 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-732
7307: 0.0734207034111023 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-733
7317: 0.05348915606737137 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-734
7327: 0.08413299918174744 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-735
7337: 0.17319455742835999 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-736
7347: 0.20971857011318207 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-737
7357: 0.04564150422811508 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-738
7367: 0.21635885536670685 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-739
7377: 0.14933565258979797 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-740
7387: 0.13964515924453735 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-741
7397: 0.06635533273220062 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-742
7407: 0.25247281789779663 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-743
7417: 0.07755842804908752 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-744
7427: 0.1271463930606842 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-745
7437: 0.0624089390039444 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-746
7447: 0.25830957293510437 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-747
7457: 0.10198751091957092 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-748
7467: 0.21774950623512268 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-749
7477: 0.20809400081634521 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-750
7487: 0.19532154500484467 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-751
7497: 0.14811763167381287 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-752
Epoch 3 terminated
Training accuracy: 0.9349720478057861
7502: 0.05615463852882385 - accuracy: 1.0
Checkpoint saved: ./tf_ckpts/ckpt-753
7512: 0.15803706645965576 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-754
7522: 0.08983398973941803 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-755
7532: 0.2037683129310608 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-756
7542: 0.12219894677400589 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-757
7552: 0.2445196658372879 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-758
7562: 0.1374882459640503 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-759
7572: 0.16109740734100342 - accuracy: 0.921875
Checkpoint saved: ./tf_ckpts/ckpt-760
7582: 0.10966391861438751 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-761
7592: 0.10664992779493332 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-762
7602: 0.20183879137039185 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-763
7612: 0.09861426055431366 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-764
7622: 0.17424529790878296 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-765
7632: 0.060265325009822845 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-766
7642: 0.19268208742141724 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-767
7652: 0.02474464289844036 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-768
7662: 0.1340375542640686 - accuracy: 0.9156249761581421
Checkpoint saved: ./tf_ckpts/ckpt-769
7672: 0.09903974831104279 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-770
7682: 0.09782516956329346 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-771
7692: 0.2042495608329773 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-772
7702: 0.09567180275917053 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-773
7712: 0.15777382254600525 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-774
7722: 0.06971020251512527 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-775
7732: 0.11647143959999084 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-776
7742: 0.05462883785367012 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-777
7752: 0.1805456429719925 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-778
7762: 0.14694225788116455 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-779
7772: 0.09295820444822311 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-780
7782: 0.10785957425832748 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-781
7792: 0.11117737740278244 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-782
7802: 0.12057673931121826 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-783
7812: 0.023850537836551666 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-784
7822: 0.16215834021568298 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-785
7832: 0.07241997867822647 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-786
7842: 0.061296090483665466 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-787
7852: 0.04074087366461754 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-788
7862: 0.19633394479751587 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-789
7872: 0.07931050658226013 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-790
7882: 0.058098431676626205 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-791
7892: 0.19519886374473572 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-792
7902: 0.05463628098368645 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-793
7912: 0.3652750849723816 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-794
7922: 0.16027680039405823 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-795
7932: 0.0941164493560791 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-796
7942: 0.08148390054702759 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-797
7952: 0.2816261947154999 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-798
7962: 0.08738593757152557 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-799
7972: 0.1824023425579071 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-800
7982: 0.013850651681423187 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-801
7992: 0.0628540962934494 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-802
8002: 0.2539260685443878 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-803
8012: 0.12765131890773773 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-804
8022: 0.09192459285259247 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-805
8032: 0.3189871907234192 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-806
8042: 0.1904374659061432 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-807
8052: 0.1066729798913002 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-808
8062: 0.16894075274467468 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-809
8072: 0.18593630194664001 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-810
8082: 0.09176237881183624 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-811
8092: 0.12359866499900818 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-812
8102: 0.17641712725162506 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-813
8112: 0.13498461246490479 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-814
8122: 0.22081105411052704 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-815
8132: 0.08629455417394638 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-816
8142: 0.5269008874893188 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-817
8152: 0.19363921880722046 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-818
8162: 0.051725566387176514 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-819
8172: 0.14458341896533966 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-820
8182: 0.09895344078540802 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-821
8192: 0.19490951299667358 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-822
8202: 0.1864694058895111 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-823
8212: 0.1283525675535202 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-824
8222: 0.18383201956748962 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-825
8232: 0.08013293147087097 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-826
8242: 0.11226821690797806 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-827
8252: 0.27015870809555054 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-828
8262: 0.11440896987915039 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-829
8272: 0.07979097962379456 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-830
8282: 0.2096458077430725 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-831
8292: 0.011804923415184021 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-832
8302: 0.15468153357505798 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-833
8312: 0.1660299003124237 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-834
8322: 0.13683411478996277 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-835
8332: 0.028885768726468086 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-836
8342: 0.07452739030122757 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-837
8352: 0.14540156722068787 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-838
8362: 0.10284852981567383 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-839
8372: 0.13823667168617249 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-840
8382: 0.1216227114200592 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-841
8392: 0.20081380009651184 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-842
8402: 0.09980437159538269 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-843
8412: 0.058125000447034836 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-844
8422: 0.0920974463224411 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-845
8432: 0.04998517408967018 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-846
8442: 0.20738810300827026 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-847
8452: 0.09953831881284714 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-848
8462: 0.157232403755188 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-849
8472: 0.039903998374938965 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-850
8482: 0.03358286991715431 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-851
8492: 0.17175862193107605 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-852
8502: 0.30264055728912354 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-853
8512: 0.06497126072645187 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-854
8522: 0.11125406622886658 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-855
8532: 0.04844195395708084 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-856
8542: 0.12663647532463074 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-857
8552: 0.06659144908189774 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-858
8562: 0.06379753351211548 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-859
8572: 0.029674090445041656 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-860
8582: 0.06660765409469604 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-861
8592: 0.2106027603149414 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-862
8602: 0.039481692016124725 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-863
8612: 0.2772105634212494 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-864
8622: 0.14839240908622742 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-865
8632: 0.2453489899635315 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-866
8642: 0.251103937625885 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-867
8652: 0.12604284286499023 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-868
8662: 0.12361596524715424 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-869
8672: 0.02854461781680584 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-870
8682: 0.0967574268579483 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-871
8692: 0.07123518735170364 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-872
8702: 0.21585586667060852 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-873
8712: 0.10995791852474213 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-874
8722: 0.22088921070098877 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-875
8732: 0.13559094071388245 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-876
8742: 0.04480748996138573 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-877
8752: 0.08445419371128082 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-878
8762: 0.08194977045059204 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-879
8772: 0.022391486912965775 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-880
8782: 0.14867913722991943 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-881
8792: 0.08633451163768768 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-882
8802: 0.0865282416343689 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-883
8812: 0.12253574281930923 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-884
8822: 0.11390109360218048 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-885
8832: 0.08518610894680023 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-886
8842: 0.1660052239894867 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-887
8852: 0.020518193021416664 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-888
8862: 0.07225710898637772 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-889
8872: 0.14816416800022125 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-890
8882: 0.1628066748380661 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-891
8892: 0.07917086780071259 - accuracy: 0.925000011920929
Checkpoint saved: ./tf_ckpts/ckpt-892
8902: 0.09065921604633331 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-893
8912: 0.10363924503326416 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-894
8922: 0.03429902344942093 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-895
8932: 0.10593457520008087 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-896
8942: 0.07599122822284698 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-897
8952: 0.03557657450437546 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-898
8962: 0.10238602012395859 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-899
8972: 0.1921713650226593 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-900
8982: 0.17381834983825684 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-901
8992: 0.020454425364732742 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-902
9002: 0.22798535227775574 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-903
9012: 0.1812557727098465 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-904
9022: 0.09842175990343094 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-905
9032: 0.09545670449733734 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-906
9042: 0.14455246925354004 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-907
9052: 0.03376895934343338 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-908
9062: 0.07530423998832703 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-909
9072: 0.06803298741579056 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-910
9082: 0.10564328730106354 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-911
9092: 0.19487369060516357 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-912
9102: 0.0761404037475586 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-913
9112: 0.2053276151418686 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-914
9122: 0.1899620145559311 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-915
9132: 0.09852288663387299 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-916
9142: 0.1949181705713272 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-917
9152: 0.09478764235973358 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-918
9162: 0.05544209107756615 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-919
9172: 0.09601350873708725 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-920
9182: 0.07367692142724991 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-921
9192: 0.08057829737663269 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-922
9202: 0.07338651269674301 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-923
9212: 0.14284221827983856 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-924
9222: 0.19822576642036438 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-925
9232: 0.04300424084067345 - accuracy: 0.9312499761581421
Checkpoint saved: ./tf_ckpts/ckpt-926
9242: 0.14636573195457458 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-927
9252: 0.09019359946250916 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-928
9262: 0.03002508543431759 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-929
9272: 0.03748044744133949 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-930
9282: 0.20693276822566986 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-931
9292: 0.0824795737862587 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-932
9302: 0.08039450645446777 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-933
9312: 0.09013963490724564 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-934
9322: 0.11994315683841705 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-935
9332: 0.058056097477674484 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-936
9342: 0.16931864619255066 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-937
9352: 0.21335694193840027 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-938
9362: 0.2739473879337311 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-939
9372: 0.09714039415121078 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-940
Epoch 4 terminated
Training accuracy: 0.9417409300804138
9377: 0.03430808335542679 - accuracy: 1.0
Checkpoint saved: ./tf_ckpts/ckpt-941
9387: 0.07953767478466034 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-942
9397: 0.02235575206577778 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-943
9407: 0.14430072903633118 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-944
9417: 0.06751185655593872 - accuracy: 0.9281250238418579
Checkpoint saved: ./tf_ckpts/ckpt-945
9427: 0.18757683038711548 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-946
9437: 0.1620183289051056 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-947
9447: 0.16848142445087433 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-948
9457: 0.0665828138589859 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-949
9467: 0.07934942841529846 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-950
9477: 0.0334346741437912 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-951
9487: 0.06634168326854706 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-952
9497: 0.07218683511018753 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-953
9507: 0.05416037514805794 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-954
9517: 0.18845710158348083 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-955
9527: 0.0824231207370758 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-956
9537: 0.16055075824260712 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-957
9547: 0.017394820228219032 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-958
9557: 0.08827725797891617 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-959
9567: 0.12261229753494263 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-960
9577: 0.11964259296655655 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-961
9587: 0.041459787636995316 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-962
9597: 0.10225985199213028 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-963
9607: 0.04447758570313454 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-964
9617: 0.0198507197201252 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-965
9627: 0.14330515265464783 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-966
9637: 0.06460915505886078 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-967
9647: 0.05985059589147568 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-968
9657: 0.11643686890602112 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-969
9667: 0.10680528730154037 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-970
9677: 0.07478970289230347 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-971
9687: 0.05834358185529709 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-972
9697: 0.1501987725496292 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-973
9707: 0.009263064712285995 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-974
9717: 0.020944509655237198 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-975
9727: 0.06545614451169968 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-976
9737: 0.11367706209421158 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-977
9747: 0.0669625923037529 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-978
9757: 0.033898018300533295 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-979
9767: 0.12171157449483871 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-980
9777: 0.03331957384943962 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-981
9787: 0.27848321199417114 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-982
9797: 0.07728947699069977 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-983
9807: 0.0407680906355381 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-984
9817: 0.06734234094619751 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-985
9827: 0.14104007184505463 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-986
9837: 0.06468687206506729 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-987
9847: 0.19348832964897156 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-988
9857: 0.015996072441339493 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-989
9867: 0.04727799445390701 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-990
9877: 0.18657656013965607 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-991
9887: 0.048796191811561584 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-992
9897: 0.060312774032354355 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-993
9907: 0.299188494682312 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-994
9917: 0.16876253485679626 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-995
9927: 0.061037901788949966 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-996
9937: 0.1623862236738205 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-997
9947: 0.1537034809589386 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-998
9957: 0.0959954708814621 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-999
9967: 0.04438973218202591 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1000
9977: 0.18053022027015686 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-1001
9987: 0.07504917681217194 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1002
9997: 0.14682963490486145 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1003
10007: 0.03210141882300377 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1004
10017: 0.43694937229156494 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1005
10027: 0.08957882225513458 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1006
10037: 0.016319019719958305 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1007
10047: 0.21266648173332214 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1008
10057: 0.2799695134162903 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1009
10067: 0.09366238862276077 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1010
10077: 0.15210075676441193 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1011
10087: 0.03864249959588051 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1012
10097: 0.12439405918121338 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1013
10107: 0.09027974307537079 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-1014
10117: 0.12266653776168823 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1015
10127: 0.24914172291755676 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1016
10137: 0.16129866242408752 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1017
10147: 0.10976549237966537 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1018
10157: 0.10885336995124817 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1019
10167: 0.022347677499055862 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1020
10177: 0.25071412324905396 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1021
10187: 0.4421924352645874 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1022
10197: 0.12595254182815552 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1023
10207: 0.031028037890791893 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1024
10217: 0.05058334395289421 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1025
10227: 0.07986287027597427 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1026
10237: 0.04832927882671356 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1027
10247: 0.07219434529542923 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1028
10257: 0.12933769822120667 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1029
10267: 0.14011293649673462 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1030
10277: 0.042089562863111496 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1031
10287: 0.03046371415257454 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1032
10297: 0.056220609694719315 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1033
10307: 0.09544138610363007 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1034
10317: 0.24695263803005219 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1035
10327: 0.10310369729995728 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1036
10337: 0.11744393408298492 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1037
10347: 0.06155266612768173 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1038
10357: 0.0399605892598629 - accuracy: 0.934374988079071
Checkpoint saved: ./tf_ckpts/ckpt-1039
10367: 0.15674754977226257 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1040
10377: 0.33460336923599243 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1041
10387: 0.019915297627449036 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1042
10397: 0.04402925819158554 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1043
10407: 0.05206295847892761 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1044
10417: 0.08219458907842636 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1045
10427: 0.011776571162045002 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1046
10437: 0.03450515493750572 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1047
10447: 0.041987113654613495 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1048
10457: 0.07871321588754654 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1049
10467: 0.12260156869888306 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1050
10477: 0.08378349244594574 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1051
10487: 0.14607809484004974 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1052
10497: 0.13055630028247833 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1053
10507: 0.2779969871044159 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1054
10517: 0.12190164625644684 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1055
10527: 0.10269652307033539 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1056
10537: 0.09823034703731537 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1057
10547: 0.02483377419412136 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1058
10557: 0.04660347104072571 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-1059
10567: 0.030562706291675568 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1060
10577: 0.10786828398704529 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1061
10587: 0.11660937964916229 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1062
10597: 0.1765322983264923 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1063
10607: 0.3329489529132843 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1064
10617: 0.17042791843414307 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1065
10627: 0.014185642823576927 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1066
10637: 0.03148447722196579 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1067
10647: 0.013131912797689438 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1068
10657: 0.07838110625743866 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1069
10667: 0.10369867831468582 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1070
10677: 0.07659350335597992 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1071
10687: 0.24124085903167725 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1072
10697: 0.14363765716552734 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-1073
10707: 0.054093457758426666 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1074
10717: 0.1658390909433365 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1075
10727: 0.023331010714173317 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1076
10737: 0.02074185200035572 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1077
10747: 0.2473219931125641 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1078
10757: 0.1852475106716156 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1079
10767: 0.03379400819540024 - accuracy: 0.9437500238418579
Checkpoint saved: ./tf_ckpts/ckpt-1080
10777: 0.12602423131465912 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1081
10787: 0.09225362539291382 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1082
10797: 0.02593677118420601 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1083
10807: 0.06061145290732384 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1084
10817: 0.1296202689409256 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1085
10827: 0.008758293464779854 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1086
10837: 0.03852803260087967 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1087
10847: 0.13848842680454254 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1088
10857: 0.06740613281726837 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1089
10867: 0.015311062335968018 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1090
10877: 0.2790069282054901 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1091
10887: 0.06237286701798439 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1092
10897: 0.24896419048309326 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1093
10907: 0.09681287407875061 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1094
10917: 0.16672834753990173 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1095
10927: 0.06335621327161789 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1096
10937: 0.08642790466547012 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1097
10947: 0.04898395016789436 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1098
10957: 0.033351778984069824 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1099
10967: 0.09200039505958557 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1100
10977: 0.057531800121068954 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1101
10987: 0.11664389073848724 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1102
10997: 0.2631675601005554 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1103
11007: 0.03313073515892029 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1104
11017: 0.09311658889055252 - accuracy: 0.940625011920929
Checkpoint saved: ./tf_ckpts/ckpt-1105
11027: 0.02503916248679161 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1106
11037: 0.03083251230418682 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1107
11047: 0.04857686534523964 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1108
11057: 0.030616173520684242 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1109
11067: 0.13654616475105286 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1110
11077: 0.023365579545497894 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1111
11087: 0.1418997049331665 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1112
11097: 0.16081586480140686 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1113
11107: 0.10232003778219223 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1114
11117: 0.11656171828508377 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1115
11127: 0.09935563802719116 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1116
11137: 0.018612569198012352 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1117
11147: 0.061183519661426544 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1118
11157: 0.07722742110490799 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1119
11167: 0.032359082251787186 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1120
11177: 0.0617777481675148 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1121
11187: 0.11272203177213669 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1122
11197: 0.0683688297867775 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1123
11207: 0.02058747410774231 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1124
11217: 0.10638720542192459 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1125
11227: 0.15955711901187897 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1126
11237: 0.2914808988571167 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1127
11247: 0.13678279519081116 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1128
Epoch 5 terminated
Training accuracy: 0.9400944709777832
11252: 0.035200297832489014 - accuracy: 1.0
Checkpoint saved: ./tf_ckpts/ckpt-1129
11262: 0.15609043836593628 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1130
11272: 0.03950650244951248 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1131
11282: 0.14915834367275238 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1132
11292: 0.04421667754650116 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1133
11302: 0.06129658594727516 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1134
11312: 0.11477760970592499 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1135
11322: 0.09452589601278305 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1136
11332: 0.07556600868701935 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1137
11342: 0.042131099849939346 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1138
11352: 0.038335517048835754 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1139
11362: 0.1188533753156662 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1140
11372: 0.06626704335212708 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1141
11382: 0.027340374886989594 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1142
11392: 0.170188307762146 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1143
11402: 0.046340007334947586 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1144
11412: 0.11709052324295044 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1145
11422: 0.04608546197414398 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-1146
11432: 0.1072797030210495 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1147
11442: 0.1414143592119217 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1148
11452: 0.060150645673274994 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1149
11462: 0.02047652006149292 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1150
11472: 0.05836416780948639 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1151
11482: 0.04909079521894455 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1152
11492: 0.020334981381893158 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-1153
11502: 0.16012102365493774 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1154
11512: 0.0484931506216526 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1155
11522: 0.35769590735435486 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1156
11532: 0.13751953840255737 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1157
11542: 0.061705488711595535 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1158
11552: 0.13615760207176208 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1159
11562: 0.010896509513258934 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1160
11572: 0.10979345440864563 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1161
11582: 0.025290781632065773 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1162
11592: 0.025715766474604607 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1163
11602: 0.06748943775892258 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1164
11612: 0.062071122229099274 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1165
11622: 0.020919397473335266 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1166
11632: 0.07932283729314804 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1167
11642: 0.10142914950847626 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1168
11652: 0.021695129573345184 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1169
11662: 0.24991968274116516 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-1170
11672: 0.058742620050907135 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1171
11682: 0.028135143220424652 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1172
11692: 0.020958632230758667 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1173
11702: 0.11885472387075424 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1174
11712: 0.04302597790956497 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1175
11722: 0.19846822321414948 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1176
11732: 0.13298873603343964 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1177
11742: 0.07830944657325745 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1178
11752: 0.101992666721344 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1179
11762: 0.22268646955490112 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1180
11772: 0.05433070659637451 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1181
11782: 0.2287016361951828 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1182
11792: 0.14201019704341888 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1183
11802: 0.044826071709394455 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1184
11812: 0.19478559494018555 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1185
11822: 0.049800097942352295 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1186
11832: 0.041889309883117676 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1187
11842: 0.021267753094434738 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1188
11852: 0.07770034670829773 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1189
11862: 0.0967961996793747 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1190
11872: 0.07752780616283417 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1191
11882: 0.07057498395442963 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1192
11892: 0.38505151867866516 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1193
11902: 0.07396605610847473 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1194
11912: 0.005938784684985876 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1195
11922: 0.17116567492485046 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1196
11932: 0.05588314309716225 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1197
11942: 0.08401913940906525 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1198
11952: 0.014679091051220894 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1199
11962: 0.08972154557704926 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1200
11972: 0.0569007508456707 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1201
11982: 0.06788323819637299 - accuracy: 0.9375
Checkpoint saved: ./tf_ckpts/ckpt-1202
11992: 0.1413070261478424 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1203
12002: 0.08613786101341248 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1204
12012: 0.05821649357676506 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1205
12022: 0.048391424119472504 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1206
12032: 0.1738101840019226 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1207
12042: 0.001900379080325365 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1208
12052: 0.10611171275377274 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1209
12062: 0.2329239845275879 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1210
12072: 0.06624791026115417 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1211
12082: 0.027568373829126358 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1212
12092: 0.02638755924999714 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1213
12102: 0.05280959978699684 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1214
12112: 0.03413724526762962 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1215
12122: 0.18049150705337524 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1216
12132: 0.08792852610349655 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1217
12142: 0.10903333127498627 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1218
12152: 0.02188262715935707 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1219
12162: 0.11359064280986786 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1220
12172: 0.015392163768410683 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1221
12182: 0.051927633583545685 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1222
12192: 0.19642317295074463 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1223
12202: 0.05770758539438248 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1224
12212: 0.1192217767238617 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1225
12222: 0.12343869358301163 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1226
12232: 0.013853052631020546 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1227
12242: 0.12172205746173859 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1228
12252: 0.14499489963054657 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1229
12262: 0.03006376139819622 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1230
12272: 0.07394077628850937 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1231
12282: 0.09474917501211166 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1232
12292: 0.068020299077034 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1233
12302: 0.05462677776813507 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1234
12312: 0.034033048897981644 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1235
12322: 0.07057851552963257 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1236
12332: 0.11370320618152618 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1237
12342: 0.18562445044517517 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1238
12352: 0.06330816447734833 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1239
12362: 0.17474821209907532 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1240
12372: 0.03376106172800064 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1241
12382: 0.1497763693332672 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1242
12392: 0.11677365005016327 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1243
12402: 0.03819810226559639 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1244
12412: 0.06315314024686813 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1245
12422: 0.021560246124863625 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1246
12432: 0.0404016487300396 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1247
12442: 0.007857101038098335 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1248
12452: 0.18331748247146606 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1249
12462: 0.2293384075164795 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1250
12472: 0.20810043811798096 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1251
12482: 0.16243934631347656 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1252
12492: 0.17214816808700562 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1253
12502: 0.11327521502971649 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1254
12512: 0.05288045108318329 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1255
12522: 0.04446658492088318 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1256
12532: 0.09215548634529114 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1257
12542: 0.05382460355758667 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1258
12552: 0.08545920252799988 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1259
12562: 0.22804851830005646 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1260
12572: 0.058264389634132385 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1261
12582: 0.0337400957942009 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1262
12592: 0.11301545798778534 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1263
12602: 0.01354717742651701 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1264
12612: 0.040513910353183746 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1265
12622: 0.04521625116467476 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1266
12632: 0.06115759164094925 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1267
12642: 0.11810468137264252 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1268
12652: 0.10870221257209778 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1269
12662: 0.07441878318786621 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1270
12672: 0.033121258020401 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1271
12682: 0.08443428575992584 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1272
12692: 0.042919378727674484 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1273
12702: 0.03492463007569313 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1274
12712: 0.009763670153915882 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1275
12722: 0.15248289704322815 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1276
12732: 0.06665145605802536 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1277
12742: 0.03677969053387642 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1278
12752: 0.23658274114131927 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1279
12762: 0.1082785427570343 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1280
12772: 0.14526838064193726 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1281
12782: 0.009873796254396439 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1282
12792: 0.03458524867892265 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1283
12802: 0.022815167903900146 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1284
12812: 0.13038477301597595 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1285
12822: 0.0651506707072258 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1286
12832: 0.02868884801864624 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1287
12842: 0.07426627725362778 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1288
12852: 0.017008280381560326 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1289
12862: 0.08340773731470108 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1290
12872: 0.054817698895931244 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1291
12882: 0.07367035746574402 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1292
12892: 0.020740093663334846 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1293
12902: 0.08825770765542984 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1294
12912: 0.05043749883770943 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1295
12922: 0.11537820845842361 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1296
12932: 0.21465006470680237 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1297
12942: 0.0922510102391243 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1298
12952: 0.06073024123907089 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1299
12962: 0.09612195938825607 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1300
12972: 0.07247776538133621 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1301
12982: 0.035263802856206894 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1302
12992: 0.04846567660570145 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1303
13002: 0.13535648584365845 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1304
13012: 0.026291877031326294 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1305
13022: 0.018043268471956253 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1306
13032: 0.12092329561710358 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1307
13042: 0.08293624222278595 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1308
13052: 0.0308503657579422 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1309
13062: 0.04443393275141716 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1310
13072: 0.07200208306312561 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1311
13082: 0.013465473428368568 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1312
13092: 0.06773550808429718 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1313
13102: 0.17070704698562622 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1314
13112: 0.2814589738845825 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1315
13122: 0.09289613366127014 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1316
Epoch 6 terminated
Training accuracy: 0.944335401058197
13127: 0.04383881017565727 - accuracy: 1.0
Checkpoint saved: ./tf_ckpts/ckpt-1317
13137: 0.027086228132247925 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1318
13147: 0.04445239529013634 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1319
13157: 0.13123610615730286 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1320
13167: 0.047837331891059875 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1321
13177: 0.06798165291547775 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1322
13187: 0.04743292182683945 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1323
13197: 0.12391474843025208 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1324
13207: 0.062470730394124985 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1325
13217: 0.026606682687997818 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1326
13227: 0.017727971076965332 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1327
13237: 0.10729353129863739 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1328
13247: 0.06320251524448395 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1329
13257: 0.017695995047688484 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1330
13267: 0.0590050108730793 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1331
13277: 0.15998950600624084 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1332
13287: 0.09228212386369705 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1333
13297: 0.10430300235748291 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1334
13307: 0.054829590022563934 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1335
13317: 0.03808126598596573 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1336
13327: 0.04535306990146637 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1337
13337: 0.019558291882276535 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1338
13347: 0.02087697945535183 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1339
13357: 0.03980139642953873 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1340
13367: 0.01917961798608303 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1341
13377: 0.11041317880153656 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1342
13387: 0.16816282272338867 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1343
13397: 0.07579264044761658 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1344
13407: 0.03948073461651802 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1345
13417: 0.028020836412906647 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1346
13427: 0.1622375100851059 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1347
13437: 0.04268515855073929 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1348
13447: 0.03611353412270546 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1349
13457: 0.0250006765127182 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1350
13467: 0.004695350304245949 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1351
13477: 0.07310712337493896 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1352
13487: 0.049465153366327286 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1353
13497: 0.09065896272659302 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1354
13507: 0.03038269653916359 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1355
13517: 0.02812555432319641 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1356
13527: 0.027691174298524857 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1357
13537: 0.14129593968391418 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1358
13547: 0.04197917878627777 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1359
13557: 0.03347828611731529 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1360
13567: 0.047506049275398254 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1361
13577: 0.020982757210731506 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1362
13587: 0.02041098289191723 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1363
13597: 0.08666378259658813 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1364
13607: 0.002571989083662629 - accuracy: 0.996874988079071
Checkpoint saved: ./tf_ckpts/ckpt-1365
13617: 0.011217189021408558 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1366
13627: 0.13664552569389343 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1367
13637: 0.012492424808442593 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1368
13647: 0.0412873700261116 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1369
13657: 0.08293452858924866 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1370
13667: 0.1569146066904068 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1371
13677: 0.031390849500894547 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1372
13687: 0.0378730371594429 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1373
13697: 0.11824080348014832 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1374
13707: 0.07746851444244385 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1375
13717: 0.08224507421255112 - accuracy: 0.9468749761581421
Checkpoint saved: ./tf_ckpts/ckpt-1376
13727: 0.10904490947723389 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1377
13737: 0.019729360938072205 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1378
13747: 0.06512598693370819 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1379
13757: 0.029780929908156395 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1380
13767: 0.2675342261791229 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1381
13777: 0.07104290276765823 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1382
13787: 0.098321832716465 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1383
13797: 0.13279929757118225 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1384
13807: 0.035093557089567184 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1385
13817: 0.02827645093202591 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1386
13827: 0.013811571523547173 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1387
13837: 0.04960101470351219 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1388
13847: 0.04130471125245094 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1389
13857: 0.037556786090135574 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1390
13867: 0.11710980534553528 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1391
13877: 0.07193898409605026 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1392
13887: 0.04281071946024895 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1393
13897: 0.04786904156208038 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1394
13907: 0.18492139875888824 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1395
13917: 0.016094809398055077 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1396
13927: 0.048682376742362976 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1397
13937: 0.20977357029914856 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1398
13947: 0.056729190051555634 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1399
13957: 0.004481831565499306 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1400
13967: 0.021043844521045685 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1401
13977: 0.03168924152851105 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1402
13987: 0.08332425355911255 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1403
13997: 0.05711216852068901 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1404
14007: 0.12792536616325378 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1405
14017: 0.019259024411439896 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1406
14027: 0.023340538144111633 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1407
14037: 0.07078862190246582 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1408
14047: 0.0271038468927145 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1409
14057: 0.027387090027332306 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1410
14067: 0.1418284773826599 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1411
14077: 0.11563695222139359 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1412
14087: 0.1497022807598114 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1413
14097: 0.13726568222045898 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1414
14107: 0.0527181550860405 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1415
14117: 0.02169588953256607 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1416
14127: 0.104857437312603 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1417
14137: 0.030957330018281937 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1418
14147: 0.05705855041742325 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1419
14157: 0.05036376789212227 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1420
14167: 0.041316211223602295 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1421
14177: 0.02777783013880253 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1422
14187: 0.007647860329598188 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1423
14197: 0.015867076814174652 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1424
14207: 0.03938378393650055 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1425
14217: 0.09965458512306213 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1426
14227: 0.04788085073232651 - accuracy: 0.996874988079071
Checkpoint saved: ./tf_ckpts/ckpt-1427
14237: 0.022611821070313454 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1428
14247: 0.03433557599782944 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1429
14257: 0.3904045522212982 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1430
14267: 0.07858184725046158 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1431
14277: 0.01198661606758833 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1432
14287: 0.046397484838962555 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1433
14297: 0.0058910660445690155 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1434
14307: 0.09883903712034225 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1435
14317: 0.006772109307348728 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1436
14327: 0.10977452248334885 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1437
14337: 0.038924939930438995 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1438
14347: 0.1549454629421234 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1439
14357: 0.12264688313007355 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1440
14367: 0.0889560654759407 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1441
14377: 0.0017314720898866653 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1442
14387: 0.1105412170290947 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1443
14397: 0.016544796526432037 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1444
14407: 0.005718627478927374 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1445
14417: 0.0627579316496849 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1446
14427: 0.12915083765983582 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1447
14437: 0.042087823152542114 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1448
14447: 0.03016076236963272 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1449
14457: 0.0773974284529686 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1450
14467: 0.04497387632727623 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1451
14477: 0.09438049793243408 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1452
14487: 0.07224141061306 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1453
14497: 0.08820266276597977 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1454
14507: 0.2881554365158081 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1455
14517: 0.05441877990961075 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1456
14527: 0.12550735473632812 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1457
14537: 0.1414155215024948 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1458
14547: 0.024779187515378 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1459
14557: 0.02048659138381481 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1460
14567: 0.08198527246713638 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1461
14577: 0.0010014460422098637 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1462
14587: 0.008287154138088226 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1463
14597: 0.1835632026195526 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1464
14607: 0.12650416791439056 - accuracy: 0.953125
Checkpoint saved: ./tf_ckpts/ckpt-1465
14617: 0.03389511629939079 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1466
14627: 0.14661338925361633 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1467
14637: 0.030368434265255928 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1468
14647: 0.03597754240036011 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1469
14657: 0.020104117691516876 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1470
14667: 0.008708802983164787 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1471
14677: 0.08062136173248291 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1472
14687: 0.022614333778619766 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1473
14697: 0.10448715090751648 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1474
14707: 0.004639206454157829 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1475
14717: 0.0926341861486435 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1476
14727: 0.01797078363597393 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1477
14737: 0.2230081409215927 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1478
14747: 0.06099401041865349 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1479
14757: 0.10503096133470535 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1480
14767: 0.04054417833685875 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1481
14777: 0.009422412142157555 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1482
14787: 0.033451639115810394 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1483
14797: 0.07905921339988708 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1484
14807: 0.005778261926025152 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1485
14817: 0.03371455520391464 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1486
14827: 0.043339841067790985 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1487
14837: 0.1315384954214096 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1488
14847: 0.05034181475639343 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1489
14857: 0.0075037917122244835 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1490
14867: 0.06212437525391579 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1491
14877: 0.08999546617269516 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1492
14887: 0.02063976600766182 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1493
14897: 0.09534354507923126 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1494
14907: 0.08194807171821594 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1495
14917: 0.03917987644672394 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1496
14927: 0.04262357950210571 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1497
14937: 0.3903382420539856 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1498
14947: 0.14491590857505798 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1499
14957: 0.012646847404539585 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1500
14967: 0.06660410761833191 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1501
14977: 0.21424615383148193 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1502
14987: 0.0599079430103302 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1503
14997: 0.10778552293777466 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1504
Epoch 7 terminated
Training accuracy: 0.9561934471130371
15002: 0.012047071009874344 - accuracy: 1.0
Checkpoint saved: ./tf_ckpts/ckpt-1505
15012: 0.04081715643405914 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1506
15022: 0.06770163774490356 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1507
15032: 0.06507143378257751 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1508
15042: 0.01026846282184124 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1509
15052: 0.07591648399829865 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1510
15062: 0.06924840807914734 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1511
15072: 0.07207047939300537 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1512
15082: 0.02737736515700817 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1513
15092: 0.03386343643069267 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1514
15102: 0.12901538610458374 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1515
15112: 0.03414120152592659 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1516
15122: 0.036130696535110474 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1517
15132: 0.02144131436944008 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1518
15142: 0.06446665525436401 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1519
15152: 0.01625201106071472 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1520
15162: 0.07786524295806885 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1521
15172: 0.13519270718097687 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1522
15182: 0.05338113754987717 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1523
15192: 0.049154751002788544 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1524
15202: 0.008236019872128963 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1525
15212: 0.1310575157403946 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1526
15222: 0.040937505662441254 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1527
15232: 0.036539897322654724 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1528
15242: 0.005138878710567951 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1529
15252: 0.024506021291017532 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1530
15262: 0.09052683413028717 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1531
15272: 0.021269408985972404 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1532
15282: 0.05380193144083023 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1533
15292: 0.028727306053042412 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1534
15302: 0.22993914783000946 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1535
15312: 0.04504419490695 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1536
15322: 0.1585896760225296 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1537
15332: 0.13523095846176147 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1538
15342: 0.010359971784055233 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1539
15352: 0.025711925700306892 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1540
15362: 0.06693662703037262 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1541
15372: 0.08792871981859207 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1542
15382: 0.01469973474740982 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1543
15392: 0.10000403970479965 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1544
15402: 0.119228795170784 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1545
15412: 0.049286745488643646 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1546
15422: 0.02446742355823517 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1547
15432: 0.007316336967051029 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1548
15442: 0.005076220259070396 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1549
15452: 0.02438213862478733 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1550
15462: 0.04851355031132698 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1551
15472: 0.06323609501123428 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1552
15482: 0.005319687537848949 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1553
15492: 0.007236845791339874 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1554
15502: 0.026363486424088478 - accuracy: 0.996874988079071
Checkpoint saved: ./tf_ckpts/ckpt-1555
15512: 0.09392139315605164 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1556
15522: 0.0015669059939682484 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1557
15532: 0.07050102949142456 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1558
15542: 0.18132810294628143 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1559
15552: 0.061440207064151764 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1560
15562: 0.056420695036649704 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1561
15572: 0.010645200498402119 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1562
15582: 0.02364608272910118 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1563
15592: 0.020192842930555344 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1564
15602: 0.15209442377090454 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1565
15612: 0.005877638701349497 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1566
15622: 0.028751539066433907 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1567
15632: 0.018926825374364853 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1568
15642: 0.49890652298927307 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1569
15652: 0.18486721813678741 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1570
15662: 0.0014153155498206615 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1571
15672: 0.16326521337032318 - accuracy: 0.956250011920929
Checkpoint saved: ./tf_ckpts/ckpt-1572
15682: 0.2837437093257904 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1573
15692: 0.24262025952339172 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1574
15702: 0.04100167751312256 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1575
15712: 0.009598702192306519 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1576
15722: 0.04904348403215408 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1577
15732: 0.07954422384500504 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1578
15742: 0.013568959198892117 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1579
15752: 0.051606353372335434 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1580
15762: 0.007377727422863245 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1581
15772: 0.04900216683745384 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1582
15782: 0.12137793749570847 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1583
15792: 0.008506720885634422 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1584
15802: 0.06506478041410446 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1585
15812: 0.13011938333511353 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1586
15822: 0.11630935966968536 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1587
15832: 0.005545864813029766 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1588
15842: 0.04215623810887337 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1589
15852: 0.010903612710535526 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1590
15862: 0.07991905510425568 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1591
15872: 0.10376393795013428 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1592
15882: 0.05219250172376633 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1593
15892: 0.0599631704390049 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1594
15902: 0.010728774592280388 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1595
15912: 0.0491340272128582 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1596
15922: 0.010398712009191513 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1597
15932: 0.021888727322220802 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1598
15942: 0.16304193437099457 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1599
15952: 0.06326593458652496 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1600
15962: 0.09912131726741791 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1601
15972: 0.051095783710479736 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1602
15982: 0.02504691295325756 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1603
15992: 0.026374394074082375 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1604
16002: 0.1309419572353363 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1605
16012: 0.03375190496444702 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1606
16022: 0.04183385893702507 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1607
16032: 0.0063161710277199745 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1608
16042: 0.04645611718297005 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1609
16052: 0.012509152293205261 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1610
16062: 0.013146054930984974 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1611
16072: 0.030908143147826195 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1612
16082: 0.006379535421729088 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1613
16092: 0.047038786113262177 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1614
16102: 0.045458726584911346 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1615
16112: 0.09048077464103699 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1616
16122: 0.03514571115374565 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1617
16132: 0.3007318377494812 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1618
16142: 0.152832493185997 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1619
16152: 0.027709444984793663 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1620
16162: 0.06735807657241821 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1621
16172: 0.011775238439440727 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1622
16182: 0.011432118713855743 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1623
16192: 0.060148708522319794 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1624
16202: 0.01817944459617138 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1625
16212: 0.057427823543548584 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1626
16222: 0.019850801676511765 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1627
16232: 0.03475509583950043 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1628
16242: 0.02457975596189499 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1629
16252: 0.03304994851350784 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1630
16262: 0.04253263399004936 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1631
16272: 0.010626548901200294 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1632
16282: 0.08842786401510239 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1633
16292: 0.06801681220531464 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1634
16302: 0.07211049646139145 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1635
16312: 0.13501191139221191 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1636
16322: 0.13999825716018677 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1637
16332: 0.03473808243870735 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1638
16342: 0.26366886496543884 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1639
16352: 0.0043928856030106544 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1640
16362: 0.011272162199020386 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1641
16372: 0.05702066421508789 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1642
16382: 0.009631331078708172 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1643
16392: 0.023818086832761765 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1644
16402: 0.01389419473707676 - accuracy: 0.996874988079071
Checkpoint saved: ./tf_ckpts/ckpt-1645
16412: 0.0029639843851327896 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1646
16422: 0.02034514956176281 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1647
16432: 0.12146031856536865 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1648
16442: 0.035629503428936005 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1649
16452: 0.013387342914938927 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1650
16462: 0.011979442089796066 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1651
16472: 0.08787737041711807 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1652
16482: 0.028673091903328896 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1653
16492: 0.00949260126799345 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1654
16502: 0.05912311002612114 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1655
16512: 0.0078951520845294 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1656
16522: 0.0036260983906686306 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1657
16532: 0.01957382634282112 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1658
16542: 0.030430518090724945 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1659
16552: 0.015074828639626503 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1660
16562: 0.040602073073387146 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1661
16572: 0.010902198031544685 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1662
16582: 0.007790185511112213 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1663
16592: 0.1391574889421463 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1664
16602: 0.046655260026454926 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1665
16612: 0.009589474648237228 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1666
16622: 0.014930184930562973 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1667
16632: 0.01209871843457222 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1668
16642: 0.07889753580093384 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1669
16652: 0.004672477021813393 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1670
16662: 0.01748328097164631 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1671
16672: 0.00982549600303173 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1672
16682: 0.0031484467908740044 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1673
16692: 0.0730205848813057 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1674
16702: 0.12520091235637665 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1675
16712: 0.08227705955505371 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1676
16722: 0.026839150115847588 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1677
16732: 0.004872514866292477 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1678
16742: 0.009940562769770622 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1679
16752: 0.03701264411211014 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1680
16762: 0.046735651791095734 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1681
16772: 0.07209014892578125 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1682
16782: 0.12118755280971527 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1683
16792: 0.09879669547080994 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1684
16802: 0.06328204274177551 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1685
16812: 0.023694805800914764 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1686
16822: 0.21639840304851532 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1687
16832: 0.08288955688476562 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1688
16842: 0.04626433551311493 - accuracy: 0.949999988079071
Checkpoint saved: ./tf_ckpts/ckpt-1689
16852: 0.17428748309612274 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1690
16862: 0.17884862422943115 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1691
16872: 0.05002972483634949 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1692
Epoch 8 terminated
Training accuracy: 0.9596527218818665
16877: 0.03776711970567703 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1693
16887: 0.035153087228536606 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1694
16897: 0.00778548326343298 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1695
16907: 0.1018223911523819 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1696
16917: 0.017072437331080437 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1697
16927: 0.13189145922660828 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1698
16937: 0.05056905373930931 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1699
16947: 0.06064844876527786 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1700
16957: 0.04185695946216583 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1701
16967: 0.031233537942171097 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1702
16977: 0.05237242579460144 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1703
16987: 0.14070025086402893 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1704
16997: 0.020064260810613632 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1705
17007: 0.01578482799232006 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1706
17017: 0.1421477496623993 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1707
17027: 0.018988996744155884 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1708
17037: 0.019121255725622177 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1709
17047: 0.0019574621692299843 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1710
17057: 0.06483152508735657 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1711
17067: 0.05623863264918327 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1712
17077: 0.06554731726646423 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1713
17087: 0.0026686550118029118 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1714
17097: 0.012836441397666931 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1715
17107: 0.03407450392842293 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1716
17117: 0.019316043704748154 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1717
17127: 0.023050330579280853 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1718
17137: 0.07176010310649872 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1719
17147: 0.018788035959005356 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1720
17157: 0.0513279065489769 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1721
17167: 0.015236752107739449 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1722
17177: 0.07683687657117844 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1723
17187: 0.007710565812885761 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1724
17197: 0.04917996749281883 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1725
17207: 0.025218429043889046 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1726
17217: 0.05917363613843918 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1727
17227: 0.024073023349046707 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1728
17237: 0.019222727045416832 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1729
17247: 0.036508046090602875 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1730
17257: 0.006141097284853458 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1731
17267: 0.049863483756780624 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1732
17277: 0.03698773309588432 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1733
17287: 0.15843094885349274 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1734
17297: 0.011011759750545025 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1735
17307: 0.007463130168616772 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1736
17317: 0.02663794904947281 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1737
17327: 0.024944806471467018 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1738
17337: 0.055101215839385986 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1739
17347: 0.1245468258857727 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1740
17357: 0.00567619688808918 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1741
17367: 0.032839223742485046 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1742
17377: 0.1677781492471695 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1743
17387: 0.0009186452371068299 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1744
17397: 0.005071697756648064 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1745
17407: 0.26516956090927124 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1746
17417: 0.04321247339248657 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1747
17427: 0.0346556231379509 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1748
17437: 0.08411053568124771 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1749
17447: 0.013345139101147652 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1750
17457: 0.03398881480097771 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1751
17467: 0.0029978721868246794 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1752
17477: 0.22683656215667725 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1753
17487: 0.020715726539492607 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1754
17497: 0.23517446219921112 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1755
17507: 0.0028473539277911186 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1756
17517: 0.2247457355260849 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1757
17527: 0.02683962509036064 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1758
17537: 0.01503288745880127 - accuracy: 1.0
Checkpoint saved: ./tf_ckpts/ckpt-1759
17547: 0.06738875806331635 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1760
17557: 0.526520848274231 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1761
17567: 0.015891138464212418 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1762
17577: 0.0028963538352400064 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1763
17587: 0.00745365908369422 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1764
17597: 0.27169710397720337 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1765
17607: 0.039804041385650635 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1766
17617: 0.0610320158302784 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1767
17627: 0.12347796559333801 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1768
17637: 0.03692493960261345 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1769
17647: 0.040966473519802094 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1770
17657: 0.06012927368283272 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1771
17667: 0.00839992519468069 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1772
17677: 0.04812886193394661 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1773
17687: 0.05461379140615463 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1774
17697: 0.0350489541888237 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1775
17707: 0.005967522040009499 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1776
17717: 0.004626818932592869 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1777
17727: 0.007814260199666023 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1778
17737: 0.03802026808261871 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1779
17747: 0.13574430346488953 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1780
17757: 0.08397725224494934 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1781
17767: 0.03659665212035179 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1782
17777: 0.046806201338768005 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1783
17787: 0.01577318087220192 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1784
17797: 0.007969637401401997 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1785
17807: 0.15869857370853424 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1786
17817: 0.123809814453125 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1787
17827: 0.06416579335927963 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1788
17837: 0.008174638263881207 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1789
17847: 0.03223167732357979 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1790
17857: 0.012714327313005924 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1791
17867: 0.018762968480587006 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1792
17877: 0.04852355644106865 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1793
17887: 0.004401304759085178 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1794
17897: 0.014734065160155296 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1795
17907: 0.1275779902935028 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1796
17917: 0.08525779098272324 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1797
17927: 0.004384136758744717 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1798
17937: 0.02550223097205162 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1799
17947: 0.01889936253428459 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1800
17957: 0.09821644425392151 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1801
17967: 0.03943337872624397 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1802
17977: 0.01532632764428854 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1803
17987: 0.033894263207912445 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1804
17997: 0.012227747589349747 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1805
18007: 0.25701993703842163 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1806
18017: 0.16059356927871704 - accuracy: 0.965624988079071
Checkpoint saved: ./tf_ckpts/ckpt-1807
18027: 0.007612513843923807 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1808
18037: 0.059369493275880814 - accuracy: 0.9624999761581421
Checkpoint saved: ./tf_ckpts/ckpt-1809
18047: 0.004248545505106449 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1810
18057: 0.11768467724323273 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1811
18067: 0.0376955084502697 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1812
18077: 0.012881155125796795 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1813
18087: 0.24142758548259735 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1814
18097: 0.021990835666656494 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1815
18107: 0.4334315061569214 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1816
18117: 0.04578471928834915 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1817
18127: 0.00632237084209919 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1818
18137: 0.025395844131708145 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1819
18147: 0.06922250241041183 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1820
18157: 0.032624561339616776 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1821
18167: 0.0545964352786541 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1822
18177: 0.06974072754383087 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1823
18187: 0.03325891122221947 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1824
18197: 0.05830732360482216 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1825
18207: 0.02614925242960453 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1826
18217: 0.02996884100139141 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1827
18227: 0.0037021213211119175 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1828
18237: 0.003330634441226721 - accuracy: 0.996874988079071
Checkpoint saved: ./tf_ckpts/ckpt-1829
18247: 0.03151363879442215 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1830
18257: 0.020342448726296425 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1831
18267: 0.001363527961075306 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1832
18277: 0.1046261265873909 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1833
18287: 0.03850431367754936 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1834
18297: 0.014210526831448078 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1835
18307: 0.017285166308283806 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1836
18317: 0.10776480287313461 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1837
18327: 0.0318780243396759 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1838
18337: 0.024223918095231056 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1839
18347: 0.12676215171813965 - accuracy: 0.9593750238418579
Checkpoint saved: ./tf_ckpts/ckpt-1840
18357: 0.012522341683506966 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1841
18367: 0.048014767467975616 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1842
18377: 0.1485043168067932 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1843
18387: 0.022757528349757195 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1844
18397: 0.005224776919931173 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1845
18407: 0.14579184353351593 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1846
18417: 0.023466099053621292 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1847
18427: 0.10129988193511963 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1848
18437: 0.27249079942703247 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1849
18447: 0.08925662934780121 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1850
18457: 0.023936893790960312 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1851
18467: 0.037125732749700546 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1852
18477: 0.09906131774187088 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1853
18487: 0.02074611559510231 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1854
18497: 0.3503255248069763 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1855
18507: 0.0027906482573598623 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1856
18517: 0.03344731777906418 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1857
18527: 0.003694287035614252 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1858
18537: 0.005576401483267546 - accuracy: 0.9781249761581421
Checkpoint saved: ./tf_ckpts/ckpt-1859
18547: 0.01923177018761635 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1860
18557: 0.03376153111457825 - accuracy: 0.984375
Checkpoint saved: ./tf_ckpts/ckpt-1861
18567: 0.01228055078536272 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1862
18577: 0.009237004444003105 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1863
18587: 0.05241063982248306 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1864
18597: 0.0284521896392107 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1865
18607: 0.0037587000988423824 - accuracy: 0.987500011920929
Checkpoint saved: ./tf_ckpts/ckpt-1866
18617: 0.12636534869670868 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1867
18627: 0.07733973860740662 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1868
18637: 0.056018661707639694 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1869
18647: 0.0030069751664996147 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1870
18657: 0.016612861305475235 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1871
18667: 0.05993301421403885 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1872
18677: 0.013974313624203205 - accuracy: 0.9937499761581421
Checkpoint saved: ./tf_ckpts/ckpt-1873
18687: 0.007083300035446882 - accuracy: 0.9906250238418579
Checkpoint saved: ./tf_ckpts/ckpt-1874
18697: 0.10805588960647583 - accuracy: 0.9750000238418579
Checkpoint saved: ./tf_ckpts/ckpt-1875
18707: 0.12039343267679214 - accuracy: 0.971875011920929
Checkpoint saved: ./tf_ckpts/ckpt-1876
18717: 0.14748132228851318 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1877
18727: 0.17064547538757324 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1878
18737: 0.09717033058404922 - accuracy: 0.96875
Checkpoint saved: ./tf_ckpts/ckpt-1879
18747: 0.03381621837615967 - accuracy: 0.981249988079071
Checkpoint saved: ./tf_ckpts/ckpt-1880
Epoch 9 terminated
Training accuracy: 0.9567422866821289
```
