
<!-- <div align="center"> -->

<!-- <h1>AutoStory</h1> -->


<p align="center">

  <h2 align="center">AutoStory: Generating Diverse Storytelling Images with Minimal Human Effort</h2>
  <p align="center">
    <a href="https://github.com/encounter1997"><strong>Wen Wang*</strong></a>
    ·
    <a href="https://github.com/volcverse"><strong>Canyu Zhao*</strong></a>
    ·  
    <a href="https://scholar.google.com/citations?user=FaOqRpcAAAAJ"><strong>Hao Chen</strong></a>
    ·
    <a href="https://github.com/Aziily"><strong>Zhekai Chen</strong></a>
    ·
    <a href="https://zkcys001.github.io/"><strong>Kecheng Zheng</strong></a>
    ·
    <a href="https://cshen.github.io/"><strong>Chunhua Shen</strong></a>
    <br>
    Zhejiang University
    <br>
    </br>
        <a href="https://arxiv.org/abs/2311.11243">
        <img src='https://img.shields.io/badge/arxiv-AutoStory-blue' alt='Paper PDF'></a>
        <a href="https://aim-uofa.github.io/AutoStory/">
        <img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
  </p>
</p>


<p align="center"><b>Code will be released soon!</b></p>

Story visualization aims to generate a series of images that match the story described in texts, and it requires the generated images to satisfy high quality, alignment with the text description, and consistency in character identities. Given the complexity of story visualization, existing methods drastically simplify the problem by considering only a few specific characters and scenarios, or requiring the users to provide per-image control conditions such as sketches. However, these simplifications render these methods incompetent for real applications. 
To this end, we propose an automated story visualization system that can effectively generate diverse, high-quality, and consistent sets of story images, with minimal human interactions. Specifically, we utilize the comprehension and planning capabilities of large language models for layout planning, and then leverage large-scale text-to-image models to generate sophisticated story images based on the layout. We empirically find that sparse control conditions, such as bounding boxes, are suitable for layout planning, while dense control conditions, e.g., sketches, and keypoints, are suitable for generating high-quality image content. To obtain the best of both worlds, we devise a dense condition generation module to transform simple bounding box layouts into sketch or keypoint control conditions for final image generation, which not only improves the image quality but also allows easy and intuitive user interactions. 
In addition, we propose a simple yet effective method to generate multi-view consistent character images, eliminating the reliance on human labor to collect or draw character images. This allows our method to obtain consistent story visualization even when only texts are provided as input. Both qualitative and quantitative experiments demonstrate the superiority of our method.


## Results

<image src="docs/results.png" />



## Citing
If you find our work useful, please consider citing:


```BibTeX
@article{AutoStory,
  title={AutoStory: Generating Diverse Storytelling Images with Minimal Human Effort},
  author={Wang, Wen and Zhao, Canyu and Chen, Hao and Chen, Zhekai and Zheng, Kecheng and Shen, Chunhua},
  publisher={arXiv preprint arXiv:2311.11243},
  year={2023},
}
```
