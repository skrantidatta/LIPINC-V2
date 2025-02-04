# Detecting Lip-Syncing Deepfakes: Vision Temporal Transformer for Analyzing Mouth Inconsistencies 
Soumyya Kanti Datta, Shan Jia, Siwei Lyu

Paper
## Abstract
Deepfakes are AI-generated media in which the original content is digitally altered to create convincing but manipulated images, videos, or audio. Among the various types of deepfakes, lip-syncing deepfakes are one of the most challenging deepfakes to detect. In these videos, a person's lip movements are synthesized to match altered or entirely new audio using AI models. Therefore, unlike other types of deepfakes, the artifacts in lip-syncing deepfakes are confined to the mouth region, making them more subtle and, thus harder to discern. In this paper, we introduce a novel approach utilizing a combination of Vision Temporal Transformer with multihead cross-attention to detect lip-syncing deepfakes by identifying spatiotemporal inconsistencies in the mouth region. These inconsistencies appear across adjacent frames and persist throughout the video. Our model can successfully capture these subtle spatiotemporal irregularities, achieving state-of-the-art performance on several benchmark deepfake datasets. Additionally, we created a new lip-syncing deepfake dataset, LipSyncTIMIT, which was generated using five state-of-the-art lip-syncing models.

<img src='./Images/LIPINCV2.png' width=900>


# LipSyncTimit Dataset : A Multimodal Dataset for LipSyncing DeepFake Forensics

LipSyncTimit Dataset includes 202 real videos from the VidTIMIT dataset, which contains recordings of 43 volunteers reciting short sentences, with synchronized audio and visual tracks. We collected real audio samples from the LSR2 dataset, and AI-generated (fake) audio samples were obtained from the LibriSeVoc dataset. We also created two compressed versions of our dataset using constant rate factors of 23 and 40. The lip-sync deepfakes were created using five state-of-the-art forgery methods. In total, our dataset includes 9,090 lip-syncing deepfake videos, along with their compressed versions

<img src='./Images/LipSyncTIMIT1.png' width=900>

## Download

If you would like to access the LipSyncTimit Dataset, please fill out this [google form](https://docs.google.com/forms/d/e/1FAIpQLSeKn-OAlJKcOZTU1k6GXVZZjkIuHbGs3am9ScvqkKE7M35psA/viewform?usp=sharing) . The download link will be sent to you once the form is accepted. If you have any questions, please send email to soumyyak@buffalo.edu

## Dataset Structure
Please refer to our paper for details.
```
LipSyncTimit Dataset
|-- Original Size # 3,232 videos of original size
  |-- RealVideo # 202 Real videos from VidTIMIT dataset
  |-- FakeVideo-OriginalAudio # 1010 Lip-syncing deepfake videos generated using the audio from the real videos in the VidTIMIT dataset to manipulate the real videos from the same dataset.
    |-- Diff2Lip # 202 Lip-syncing deepfake videos generated using Diff2Lip model.
    |-- Video_Retalking # 202 Lip-syncing deepfake videos generated using Video_Retalking model.
    |-- Wav2lip # 202 Lip-syncing deepfake videos generated using Wav2lip model.
    |-- Wav2lip_GAN # 202 Lip-syncing deepfake videos generated using Wav2lip_GAN model.
    |-- IP_LAP # 202 Lip-syncing deepfake videos generated using IP_LAP model.
  |-- FakeVideo-RealAudio # 1010 Lip-syncing deepfake videos generated using the real audio from the LRS2 dataset to manipulate the real videos from the VidTIMIT dataset.
  |-- FakeVideo-FakeAudio # 1010 Lip-syncing deepfake videos generated using the AI generated(fake) audio from the LibriSeVoc dataset to manipulate the real videos from the VidTIMIT dataset.
|-- LipSyncTimit_compression23 # 3,232 compressed videos with constant rate factors of 23
|-- LipSyncTimit_compression40 # 3,232 compressed videos with constant rate factors of 40
```
## Privacy Statement
This dataset is released under the Terms to Use Celeb-DF, which is provided "as it is" and we are not responsible for any subsequence from using this dataset. All original videos of the Celeb-DF dataset are obtained from the Internet which are not property of the authors or the authors’ affiliated institutions. Neither the authors or the authors’ affiliated institution are responsible for the content nor the meaning of these videos. If you feel uncomfortable about your identity shown in this dataset, please contact us and we will remove corresponding information from the dataset.

# Prerequisites
- `Python 3.10` 
- Install necessary packages using `pip install -r requirements.txt`.
- Download the dlib’s pre-trained facial landmark detector from [here](https://drive.google.com/file/d/1-Uc2rH1tiKZEh9NwmgmBFZT_6xDvGBSD/view?usp=sharing) and put it in the same folder as demo.py.
- The input video should have the face of **only 1 subject** in the entire video.
- The input video should have **1 face per frame**.

# Inference: Lip-syncing videos detection using the pre-trained models 

The input_video should be in mp4 format :
```
python demo.py --input_path {input_video_path} --output_path {output_path}
```
The demo video is saved (by default) as `{input_video_name}_demo.mp4`.


# Citation
Please cite our paper in your publications if you use our LIPINC-V2 Detection model or our LipSyncTimit dataset in your research:
```
@article{datta2025Detecting,
  title={Detecting Lip-Syncing Deepfakes: Vision Temporal Transformer for Analyzing Mouth Inconsistencies},
  author={Datta, Soumyya Kanti and Jia, Shan and Lyu, Siwei},
  year={2025}
}
```



