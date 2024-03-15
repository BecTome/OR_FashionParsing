# OR_FashionParsing

## Get Data

You can download the data from the following links (Fashionpedia 2020):

[Training data](https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip)
[Validation and Test data](https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip)


### Get Semantic Segmentation of Fashion Images

[Train Instance Attributes](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json)
[Validation Instance Attributes](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json)

**Easiest way to get the data is to use the following command:**

```bash
python get_data.py
```

**Visualize masks**

Go to [notebooks/pycocoDemo.ipynb](notebooks/pycocoDemo.ipynb)

