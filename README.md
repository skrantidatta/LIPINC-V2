# Detecting Lip-Syncing Deepfakes: Vision Temporal Transformer for Analyzing Mouth Inconsistencies 
Soumyya Kanti Datta, Shan Jia, Siwei Lyu

### [Paper](https://arxiv.org/abs/2504.01470) 

## Abstract
Deepfakes are AI-generated media in which the original content is digitally altered to create convincing but manipulated images, videos, or audio. Among the various types of deepfakes, lip-syncing deepfakes are one of the most challenging deepfakes to detect. In these videos, a person's lip movements are synthesized to match altered or entirely new audio using AI models. Therefore, unlike other types of deepfakes, the artifacts in lip-syncing deepfakes are confined to the mouth region, making them more subtle and, thus harder to discern. In this paper, we propose LIPINC-V2, a novel detection framework that leverages a combination of vision temporal transformer with multihead cross-attention to detect lip-syncing deepfakes by identifying spatiotemporal inconsistencies in the mouth region. These inconsistencies appear across adjacent frames and persist throughout the video. Our model can successfully capture both short-term and long-term variations in mouth movement, enhancing its ability to detect these inconsistencies. Additionally,  we created a new lip-syncing deepfake dataset, LipSyncTIMIT, which was generated using five state-of-the-art lip-syncing models to simulate real-world scenarios. Extensive experiments on our proposed LipSyncTIMIT dataset and two other benchmark deepfake datasets demonstrate that our model achieves state-of-the-art performance. 

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
## License and Usage Terms
The LipSyncTIMIT dataset is for non-commercial research purposes only. All original videos and audios in the LipSyncTIMIT dataset are obtained from the VidTIMIT dataset and the LRS2 dataset, respectively. All AI-generated (fake) audios in the LipSyncTIMIT dataset are obtained from the LibriSeVoc dataset. Neither the authors nor their affiliated institutions are responsible for the content or meaning of these videos/audios. You and your affiliated institution must agree not to reproduce, duplicate, publish, copy, sell, trade, resell or exploit any portion of the videos or any derived data from the dataset for any purpose.

# Prerequisites
- `Python 3.10` 
- Install necessary packages using `pip install -r requirements.txt`.
- Download the dlib’s pre-trained facial landmark detector from [here](https://drive.google.com/file/d/1-Uc2rH1tiKZEh9NwmgmBFZT_6xDvGBSD/view?usp=sharing) and put it in the same folder as demo.py.
- The LIPINC-V2 model weights are located in the ./checkpoints folder.
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
@article{datta2025detecting,
  title={Detecting Lip-Syncing Deepfakes: Vision Temporal Transformer for Analyzing Mouth Inconsistencies},
  author={Datta, Soumyya Kanti and Jia, Shan and Lyu, Siwei},
  journal={arXiv preprint arXiv:2504.01470},
  year={2025}
}

```
# Reference
[LIPINC] S K Datta, S Jia, and S Lyu, “Exposing lip-syncing deepfakes from mouth inconsistencies,” 2024 IEEE International Conference on
Multimedia and Expo (ICME), pp. 1–6, 2024.


[VidTIMIT] C Sanderson and B C Lovell, “Multi-region probabilistic histograms for robust and scalable identity inference,” in Advances in biometrics:
Third international conference, ICB 2009, alghero, italy, june 2-5, 2009. Proceedings 3. Springer, 2009, pp. 199–208.


[LRS2] T Afouras, J S Chung, A Senior, O Vinyals, and A Zisserman, “Deep audio-visual speech recognition,” IEEE transactions on pattern analysis
and machine intelligence, vol. 44, no. 12, pp. 8717–8727, 2018.


[LibriSeVoc] C Sun, S Jia, S Hou, and S Lyu, “Ai-synthesized voice detection using neural vocoder artifacts,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2023, pp. 904–912.


[Diff2Lip] S Mukhopadhyay, S Suri, R T Gadde, and A Shrivastava, “Diff2lip: Audio conditioned diffusion models for lip-synchronization,” in Proceed-
ings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 5292–5302


[Video_Retalking] K Cheng, X Cun, Y Zhang, M Xia, F Yin, M Zhu, X Wang, J Wang, andN Wang, “Videoretalking: Audio-based lip synchronization for talking
head video editing in the wild,” in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.


[Wav2lip] KR Prajwal, R Mukhopadhyay, V P Namboodiri, and CV Jawahar, “A lip sync expert is all you need for speech to lip generation in the wild,”
in Proceedings of the 28th ACM MM, 2020, pp. 484–492


[IP-Lap] W Zhong, C Fang, Y Cai, P Wei, G Zhao, L Lin, and G Li, “Identity-preserving talking face generation with landmark and appearance priors,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9729–9738.







