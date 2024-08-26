# SMFuse
This is the code of the paper titled as "SMFuse: Two-Stage Structural Map Aware Network for Multi-focus Image Fusion".

The article is accepted by International Conference on Pattern Recognition.

# Framework
![Image](https://github.com/stywmy/SMFuse/blob/main/src/framework.png)
![Image](https://github.com/stywmy/SMFuse/blob/main/src/SMENet.png)
![Image](https://github.com/stywmy/SMFuse/blob/main/src/DMGNet.png)

# Environment

- Python 3.8.16
- torch 2.0.1
- torchvision 0.15.2
- tqdm 4.65.0

# To Train

You can run the following prompt:

```python
python train.py
```
Note: SMGNet and DMGNet need to be trained separately in the code


# To Test

In the first stage:
Please create a new test image data file, modify the corresponding path in predict.py, and then run the following prompt:

```python
python predict.py 
```

Note: Before running, it is necessary to comment out the code of the second stage. Similarly, you need to comment out irrelevant parts of the train_utils/train_and_ eval.py, src/model.py, and my_dataset.py code.

In the second stage:
Please put the structure diagram obtained in the first stage into a folder, modify the corresponding path in predict.exe, and then run the following prompt:

```python
python predict.py # Obtain a preliminary decision map
python predict_final.py # Obtain the final decision map and fused image
```

Note: Before running, it is necessary to comment out the code of the first stage. Similarly, you need to comment out irrelevant parts of the train_utils/train_and_ eval.py, src/model.py, and my_dataset.py code.

# Contact Informaiton

If you have any questions, please contact me at <tianyu_shen_jnu@163.com>.
