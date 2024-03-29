# TextNeRF
 This repository accompanies the CVPR2024 paper "TextNeRF: A Novel Scene-Text Image Synthesis Method based on Neural Radiance Fields"

# Abstract
Acquiring large-scale, well-annotated datasets is essential for training robust scene text detectors, yet the process is often resource-intensive and time-consuming. While some efforts have been made to explore the synthesis of scene text images, a notable gap remains between synthetic and authentic data. In this paper, we introduce a novel method that utilizes Neural Radiance Fields (NeRF) to model real-world scenes and emulate the data collection process by rendering images from diverse camera perspectives, enriching the variability and realism of the synthesized data. A semi-supervised learning framework is proposed to categorize semantic regions within 3D scenes, ensuring consistent labeling of text regions across various viewpoints. Our method also models the pose, and view-dependent appearance of text regions, thereby offering precise control over camera poses and significantly improving the realism of text insertion and editing within scenes. Employing our technique on real-world scenes has led to the creation of a novel scene text image dataset. Compared to other existing benchmarks, the proposed dataset is distinctive in providing not only standard annotations such as bounding boxes and transcriptions but also the information of 3D pose attributes for text regions, enabling a more detailed evaluation of the robustness of text detection algorithms. Through extensive experiments, we demonstrate the effectiveness of our proposed method in enhancing the performance of scene text detectors.

# Data
We collected a total of 438 real scenes and used our method to render and annotate the data. 
This data can be downloaded from xxx

## Data Formats
The unzipped dataset has the format shown below:
```sh
|- TextNeRF_dataset
|    |- scene_0000
|    |    |- images
|    |    |    |- 000.jpg
|    |    |    |- 001.jpg
|    |    |    |- 002.jpg
|    |    |    |- ...
|    |    |- Label.txt
|    |    |- meta.json
|    |- scene_0001
|    |    |- images
|    |    |    |- 000.jpg
|    |    |    |- 001.jpg
|    |    |    |- 002.jpg
|    |    |    |- ...
|    |    |- Label.txt
|    |    |- meta.json
|    |- scene_0002
|    |    |- images
|    |    |    |- 000.jpg
|    |    |    |- 001.jpg
|    |    |    |- 002.jpg
|    |    |    |- ...
|    |    |- Label.txt
|    |    |- meta.json
|    |...
```

## Annotations
For each scene, the synthesized image results are in the "images" folder, and the text annotation content corresponding to each image is in the "Label.txt" file. 

For the "Label.txt" file, each line is the annotation information about one image, which contains two parts:

```py
# image_path   \t   json-string (a list of dict, which can be loaded by json.loads() function)
images/xxx.jpg    [{"transcription": ..., "points": ..., "text_pose": ...}, ...]
images/xxx.jpg    [{"transcription": ..., "points": ..., "text_pose": ...}, ...]
images/xxx.jpg    [{"transcription": ..., "points": ..., "text_pose": ...}, ...]
...
```

## MetaData
In addition, we also provide some basic information for the images of each scene in the "meta.json" file, including the image size, the intrinsics and extrinsics of each rendering camera, and whether the image is stylized.
```py
{
   "images/xxx.jpg": {
        "intrinsic": ...,
        "image_wh": ...,
        "pose": ...,
        "style": ...
    },
    "images/xxx.jpg": {
        "intrinsic": ...,
        "image_wh": ...,
        "pose": ...,
        "style": ...
    },
    ...
}
```
# Method Overview


# Code
coming soon...
