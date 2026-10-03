---
title: Concrete Crack Detector
emoji: 🧱
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

EfficientNet-B0 tile classifier fine-tuned on SDNET2018 to flag cracks in photos of concrete.
Upload a photo and the app cuts it into 256x256 px tiles and outlines the tiles it flags.

Research demo, not for real structural assessment. Full write-up, evaluation and code:
see the GitHub repository `crack-detection-sdnet`.
