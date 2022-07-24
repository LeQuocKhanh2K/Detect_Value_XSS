import pandas as pd
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import cv2

data_f = pd.read_csv('./XSS_dataset.csv', encoding='utf-8-sig')
# print(data_f.head())
sentences = data_f['Sentence'].values


# print(sentences[1])

def convert_to_ascii(sentence):
    sentence_ascii = []
    for i in sentence:
        if (ord(i) < 8222):

            if (ord(i) == 8217):  # ’  :  8217
                sentence_ascii.append(134)

            if (ord(i) == 8221):  # ”  :  8221
                sentence_ascii.append(129)

            if (ord(i) == 8220):  # “  :  8220
                sentence_ascii.append(130)

            if (ord(i) == 8216):  # ‘  :  8216
                sentence_ascii.append(131)

            if (ord(i) == 8217):  # ’  :  8217
                sentence_ascii.append(132)

            if (ord(i) == 8211):  # –  :  8211
                sentence_ascii.append(133)


            if (ord(i) <= 128):
                sentence_ascii.append(ord(i))

            else:
                pass

    zer = np.zeros((10000))

    for i in range(len(sentence_ascii)):
        zer[i] = sentence_ascii[i]

    zer.shape = (100, 100)

    #     plt.plot(image)
    #     plt.show()
    return zer


array = np.zeros((len(sentences), 100, 100))

for i in range(len(sentences)):
    image = convert_to_ascii(sentences[i])

    x = np.asarray(image, dtype='float')
    image = cv2.resize(x, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    image /= 128

    #     if i==1:
    #         plt.plot(image)
    #         plt.show()
    array[i] = image

print("Input data shape : ", array.shape)
data = array.reshape(array.shape[0], 100, 100, 1)
# print(data.shape)
