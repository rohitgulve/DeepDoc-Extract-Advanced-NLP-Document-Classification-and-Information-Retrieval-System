# DeepDoc-Extract-Advanced-NLP-Document-Classification-and-Information-Retrieval-System

DeepDoc-Extract-Advanced-NLP-Document-Classification-and-Information-Retrieval-System
### Installation

requires [python](https://www.python.org/download/releases/3.0/) v3(3.6.9) to run.


Installing required packages
```sh
$ pip install -r requirement.txt
```

### Prepare Data for Model
```sh
Convert data from VOC format to COCO format

$ python /util/voc2coco.py
```

### Run Application


For running Flask API,
```sh
$ python run.py
```

#####Upload the image file:
```sh
1. http://127.0.0.1:5000/train
Hit Directly. Don't upload any file 
```


```sh
1. http://127.0.0.1:5000/prediction
Input as image or pdf file

```

#####Input
Please send input through postman

```sh
{
"file" : "abc.jpg",
}
```
#### Run on Google Colab
```sh
$ invoiceExtraction.ipynb
```

