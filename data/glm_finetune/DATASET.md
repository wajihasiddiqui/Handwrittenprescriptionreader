# GLM-OCR ShareGPT dataset

- samples: 2
- file: prescriptions.json
- images: prescription_images/
- LLaMA-Factory dataset key: prescriptions
- template: glm_ocr
- labels.csv is auto-filled by GLM-OCR (edit if needed)

Train:
  python src/llamafactory_train.py train --mode lora
