import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical

##https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html


arr = np.array(['86', '103', '91', '88', '94','94','94','94','94','94','94','70', '89', '101', '84', '105',
                '94','94','91','91','91','91','91','91','91','91','91','91','91','91','91','91','94','94','94','94',
                '94','94','100', '104', '106', '69', '85','85','85','85','85',
                '85','85','85','85','85', '118',
                '75', '94','94','94','94','94','94','94','94','94','94', '68'])
arr1 = np.array(['86', '103', '91', '858'])


le = preprocessing.LabelEncoder()

transformed_labels = le.fit_transform(arr)
transformed_labels = le.transform(arr1)


print(to_categorical(transformed_labels, le.classes_.size)[0])

