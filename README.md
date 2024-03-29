# Caption Guided Visual Question Answering

## Authors
* **Sanjeev Kumar Singh**
* **Kartik Venkataraman**

Results on VQA 2.0 (12/07/2019):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">Paper</th>
    <th class="tg-0pky">Baseline</th>
    <th class="tg-0pky">Our Approach</th>
  </tr>
  <tr>
    <td class="tg-0lax">Accuracy</td>
    <td class="tg-0lax">68.37%</td>
    <td class="tg-0lax">54.20%</td>
    <td class="tg-0lax">34.27%</td>
  </tr>
</table>

* Note that baseline is trained on 20 epochs and finetuned for 10 epochs, however our model is trained for 10 epochs and fine tuned for 2 epochs only due to limited computational resources.
## Pre-requirements
**Python Version** >= 3.5 <br>
**Pytorch Version** >= 1.2 <br>
**tqdm** <br>
**pillow** <br>

## Preprocessing
(1) mkdir data && mkdir saved_models && mkdir vqa_models <br>
(2) Download data from [here](https://drive.google.com/drive/folders/1IXTsTudZtYLqmKzsXxIZbXfCnys_Izxr?usp=sharing) and put them under ``data`` folder <br>
(3) bash tools/download.sh <br>
(4) bash tools/preprocess.sh <br>

## Training
Our training process is splits into three stages. <br>
### (1) pretraining vqa models 
``CUDA_VISIBLE_DEVICES=0,1 python train_vqa_model.py --caption_dir None --learning_rate 0.0005 --joint_weight 0.4 --caption_weight 0.1 --visual_weight 0.5 --model_type hAttn`` <br>
This command will save a vqa model under ``vqa_models`` folder. <br>
Alternatively, you can also download pretrained model from here and put it under ``vqa_models`` folder by: <br>

### (2) training and extracting captions 
``CUDA_VISIBLE_DEVICES=0,1,2 python train_caption_model.py`` <br>
This command will save a vqa-caption model under ``saved_models`` folder named by the time you execute the command.

``CUDA_VISIBLE_DEVICES=0 python extract_caption_model.py --epoch 16`` <br>
This command will help you extract captions from the 16-th epochs using beam search. You can extract captions using different epoch numbers. <br>

Alternatively, you can directly use captions generated by baseline <br>
``wget -P data http://www.cs.utexas.edu/~jialinwu/dataset/qid2caption_train.pkl`` <br>
``wget -P data http://www.cs.utexas.edu/~jialinwu/dataset/qid2caption_val.pkl`` <br>
Baseline extracts 8 set of captions using different epoch numbers and while merging them, it filtered out exactly the same captions for each QA pair.

### (3) training vqa models using generated captions 
``CUDA_VISIBLE_DEVICES=0,1 python finetune_vqa_model.py --caption_dir data/qid2caption --learning_rate 0.0005 --joint_weight 0.4 --caption_weight 0.1 --visual_weight 0.5 --model_type hAttn`` <br>

Our final model is located here. [vqa model](https://drive.google.com/file/d/1oU5SHcv-R_HMFZDFfkAErTrQO1Z1FWGc/view?usp=sharing) . Please note that unlike original model provided by author, our model is only trained on 10 epochs and the fine tuned for 2 epochs.

### (4) Evaluation
``python evalTemp.py --model vqa_models/vqa_model-best.pth --joint_weight 0.4 --caption_weight 0.1 --visual_weight 0.5 --model_type hAttn`` <br>

#### Base Code Taken from [here](https://github.com/jialinwu17/generate_captions_for_vqa)
