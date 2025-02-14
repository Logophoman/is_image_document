# is_image_document
This little repo is a lightweight python script to check wether an image probably contains a document or not based on opencv.

> Please note i built a Neural Network with +99% accuracy: [is_image_document_ai](https://github.com/Logophoman/is_image_document_ai) that you might consider of the 89% accurracy with plain openCV... 

## accuracy: 

I know that the sample is way too small to really give off a good precision metric and the 88.71% I got is probably not realistic. We could easily test on 1000s of images and also make the algorithm way better, so feel free to contribute. It's good enough for what I want -> I want to quite precisely know if there is a document or not. And I personallay don't care about the occational False Positive since I'm doing OCR afterwards. 

```
Misclassified images:
documents\document26.jpg - False Negative
images\img15.jpg - False Positive
images\img16.png - False Positive
images\img22.png - False Positive
images\img27.jpg - False Positive
images\img29.png - False Positive
images\img8.jpg - False Positive

Total Images: 62
Correctly Classified: 55
Precision: 88.71%
```

### Test data: 

I used 30 document images from an [Invoices dataset on HuggingFace](https://huggingface.co/datasets/amaye15/invoices-google-ocr/)

10 random AI generated images used from [HuggingFace](https://huggingface.co/datasets/bigdata-pw/Diffusion1B) 1-10

20 random Images from [Pixabay](https://pixabay.com/) 11-30