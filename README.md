# data-handler

handbag website img_url crawling &amp; h5py amazon handbag dataset to jpg converter python code 

## file

### jupyter notebook file

각 task에 따른 jupyter notebook file 입니다.

`1. h5py_to_jpg_convert.ipynb : 약 137000장의 amazon handbag h5py 파일을 자신의 local에 jpg 형태로 변환하여 저장시켜주는 코드입니다.`

`2. web_crawling_bag.ipynb : online handbag website의 img_url을 html에서 source를 찾아내어 list에 저장하고 이를 csv로 만들어주는 코드입니다.`

`3. web_crawling_bag_2.ipynb : online handbag website의 img_url을 html에서 source를 찾아내어 list에 저장하고 이를 csv로 만들어주는 코드입니다.(하나의 jupyter notebook file로는 가독성이 뛰어나지 않을 것 같아 분리하였습니다.)`

`4. practice_pandas.ipynb : pandas library를 이용하여 여러 data를 handling하는 코드입니다.`


### data/bag_image_url_csv

각 website로부터 가져온 img_url이 csv 형태로 저장되어 있습니다.
또한 't_'로 시작하는 csv 는 column의 형태로 저장되어있던 csv를 transposed 시킨 csv 입니다. 

## requirements

1. BeautifulSoup 

2. urllib.request 

3. pandas 

4. h5py
