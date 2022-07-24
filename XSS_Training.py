from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPool2D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
from XSS_Clean_Data import *
import time
#tạo đường dẫn thư mục lưu quá trình train để Tensorboard đánh giá
NAME = "XSS-cnn-64x2-{}".format(int(time.time()))
#gọi hàm TensorBoard từ tensorflow, sẽ có 2 tập đc lưu trong folder là train và validation
tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))
#tạo các đường dẫn để lưu kết quả sau khi train xong
figure = './figure/plot.png'
checkpoint_path = './ModelCheckpoint/epoch-{epoch:02d}-val_acc-{val_accuracy:.4f}.h5'
model_save = './model/XSS-detection-final.h5'
y = data_f['Label'].values
#sử dụng hàm train_test_split để chia tập data thành tập train và test với số lượng test là 0.2
trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.2, random_state=42)
# print(trainX.shape)
#tạo đầu mạng nơ-ron với 7 layer
model = Sequential([

    Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(100, 100, 1)),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')

])
#khởi tạo model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#xem parameter của model
model.summary()
#dùng hàm checkpoint để lưu lại những model tốt nhất trong suốt quá trình training
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max'
)
#lấy batchsize và epoch
batch_size = 64
num_epoch = 80
# model training
#model fitting với số lượng batchsize và epoch đã set
H = model.fit(trainX, trainY,
              batch_size=batch_size,
              epochs=num_epoch,
              verbose=1,
              validation_data=(testX, testY),
              callbacks=[checkpoint,tensorboard]
              )
#đánh giá model
pred = model.predict(testX)
for i in range(len(pred)):
    if pred[i] > 0.5:
        pred[i] = 1
    elif pred[i] <= 0.5:
        pred[i] = 0
true = 0
false = 0

for i in range(len(pred)):
    if pred[i] == testY[i]:
        true += 1
    else:
        false += 1
#in ra số câu đúng và sai
print("correct predicted :: ", true)
print("false prediction :: ", false)
attack = 0
benign = 0
# in ra số lượng data dùng để test
for i in range(len(testY)):
    if testY[i] == 1:
        attack += 1
    else:
        benign += 1

print("Attack data in test set :: ", attack)
print(" Benign data in test set :: ", benign)

#tạo các hàm đánh giá mô hình
def accuracy_function(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy


def precision_function(tp, fp):
    precision = tp / (tp + fp)

    return precision


def recall_function(tp, fn):
    recall = tp / (tp + fn)

    return recall


def confusion_matrix(truth, predicted):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for true, pred in zip(truth, predicted):
        if true == 1:
            if pred == true:
                true_positive += 1
            elif pred != true:
                false_negative += 1

        elif true == 0:
            if pred == true:
                true_negative += 1
            elif pred != true:
                false_positive += 1

    accuracy = accuracy_function(true_positive, true_negative, false_positive, false_negative)
    precision = precision_function(true_positive, false_positive)
    recall = recall_function(true_positive, false_negative)

    return (accuracy,
            precision,
            recall)

#in ra kết quả đánh giá
accuracy, precision, recall = confusion_matrix(testY, pred)
print(" For CNN  \n Accuracy : {0} \n Precision : {1} \n Recall : {2} \n".format(accuracy, precision, recall))
model.save(model_save)
# khởi tạo để lưu ảnh trong suốt quá trình training
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"])), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(figure)
plt.show()

# python -m tensorboard.main --logdir ./logs --port 6006