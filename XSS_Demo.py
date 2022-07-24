from tensorflow.keras.models import load_model
import cv2 #Cung cấp các mã xử lý phục vụ quy trình thị giác máy tính và Learning Machine.
import numpy as np #Thư viện toán học, làm việc với ma trận và mãng với tốc độ xử lý cao.
#dùng hàm load_model trong tensorflow để đưa model dạng h5 vào
model = load_model('./model/XSS-detection.h5')
from XSS_Clean_Data import convert_to_ascii

#tạo hàm dự đoán XSS
def predict_cross_site_script():
    repeat = True

    beautify = ''
    for i in range(20): #Lặp 20 lần dự đoán
        beautify += "="
    #tạo đầu nhập input là giá trị số hoặc chữ
    print(beautify)
    input_val = input("Give a code to check XSS attack: ") #Nhập vào giá trị cần kiểm tra.
    print(beautify)
    # nếu đầu vào - thì trả lại giá trị False
    if input_val == '0':
        repeat = False
    #nếu đúng thì code sẽ gọi hàm convert qua dạng mã ascii cho đầu vào input từ đó tách các layer phân tích
    # cho đầu mạng nơ-ron của chúng ta nhận diện
    if repeat == True: #Nếu giá trị trả về là True

        image = convert_to_ascii(input_val) #Chuyển giá trị đầu vào sang ASCII

        x = np.asarray(image, dtype='float')
        # "a" - dữ liệu đầu vào
        # "dtype" - kiểu dữ liệu được suy ra từ dữ liệu đầu vào
        image = cv2.resize(x, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        # "X" - nguồn đưa vào
        # "dsize" - kích thước mong muốn cho đầu ra
        # "interpolation=cv2.INTER_CUBIC" - Cho phép nội suy 2 chiều trên vùng lân cận 4x4 pixel
        image.shape = (1, 100, 100, 1) #Gán kích thước cho image
        image /= 128 #Ảnh với kích thước 128 pixel

        prediction = model.predict(image) #Dự đoán dự vào model
        #nếu như kết quả có độ tin cậy lớn hơn 50% thì sẽ XSS
        if prediction > 0.5:

            #         print(f"Chances of attack :  {prediction[0]*100} ")
            print(" It can be XSS attack")
        # còn nếu không thì an toàn
        else:

            #         print(f" Chances of being safe {100 - (prediction*100) }")
            print("It's seems be safe")
        #đóng gói hàm
        predict_cross_site_script()

    #nếu thoát sẽ thoát khỏi code  (ctrl+C)
    elif repeat == False:
        print(" Good Bye ")

#đóng gói toàn bộ code
if __name__ == '__main__':
    predict_cross_site_script()
