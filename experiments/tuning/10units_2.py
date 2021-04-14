RAM AT BEGINNING: 0.22319412231445312
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22776031494140625

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.28882598876953125
RAM BEFORE CLASSIFER: 2.2469100952148438
RAM AFTER CLASSIFER: 2.2469100952148438
RAM BEFORE PRE-TRAINING 2.2469100952148438
RAM AFTER PRE-TRAINING 2.263519287109375
RAM BEFORE GENERATOR: 2.263519287109375
RAM AFTER DECLARING GENERATOR: 2.263519287109375
MACs of root classifier 354400
MACs of top classifier: 2560
RAM BEFORE REPORTING: 2.263519287109375

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([10, 10, 10])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([10, 10, 10])--z100-c10)-s1827

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=10, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 2698 parameters (~0.0 million)
      of which: - learnable: 2698 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 17180 parameters (~0.0 million)
      of which: - learnable: 17180 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=10, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=10)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 3660 parameters (~0.0 million)
      of which: - learnable: 3660 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 2.263519287109375
CPU BEFORE TRAINING: (30.06, 3.25)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.266162872314453
Peak mem and init mem: 963 951
GPU BEFORE EVALUATION: (9.571428571428571, 12)
RAM BEFORE EVALUATION: 2.266162872314453
CPU BEFORE EVALUATION: (125.96, 6.05)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.5192
 - Task 2: 0.8677
 - Task 3: 0.9559
 - Task 4: 0.4781
 - Task 5: 0.9981
=> Average precision over all 5 tasks: 0.7638

=> Total training time = 69.7 seconds

RAM AT THE END: 2.266223907470703
CPU AT THE END: (127.72, 6.07)
