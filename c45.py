# -*- coding: UTF-8 -*-



import numpy as np

from PIL import Image

import cv2

import matplotlib.pyplot as plt

import msvcrt

#仅使用了以上库的基础功能


# 19029212戴心研 北京工业大学 信息学部


class edge:

    """定义edge类"""


    def __init__(self,image):


        self.im=image

        self.img=Image.open(image)

        "图片大小"
        self.size=self.img.size

        self.width = self.img.width

        self.height = self.img.height


        #图片格式
        self.mode= self.img.mode

        self.type=type(self.img)


        #预先定义

        self.gray = 1

        self.fltr = 1

        self.sobel_image = 1


    def show(self):

        "原图展示"

        print(self.size)

        print(self.mode)

        print(self.type)

        self.img.show()



    def resize(self):


        "缩小图片以减少计算量"
        out = self.img.resize((800, 600))

        out.save('resize.bmp')



    def read(self):


        # 读取图像
        imag = cv2.imread(self.im)
        # 转换为灰度图像
        gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

        a=gray

        print(a)

        self.gray=a

        print("读取完成")

        return a



    def fliter(self):

        # 计算模板的大小以及模板

        def compute(delta):

            k = round(3 * delta) * 2 + 1
            print('模的大小为:', k)

            H = np.zeros((k, k))

            k1 = (k - 1) / 2

            for i in range(k):

                for j in range(k):

                    H[i, j] = (1 / (2 * 3.14 * (delta ** 2))) * math.exp(

                        (-(i - k1) ** 2 - (j - k1) ** 2) / (2 * delta ** 2))

            k3 = [k, H]

            print(H)

            print(sum(sum(H)))

            return k3

        # 相关

        def relate(a, b, k):

            n = 0

            sum1 = np.zeros((k, k))


            for m in range(k):

                for n in range(k):

                    sum1[m, n] = a[m, n] * b[m, n]

            return sum(sum(sum1))

        # 高斯滤波

        def fil(imag, delta=0.7):

            k3 = compute(delta)

            k = k3[0]

            H = k3[1]

            k1 = (k - 1) / 2

            [a, b] = imag.shape

            k1 = int(k1)

            new1 = np.zeros((k1, b))

            new2 = np.zeros(((a + (k - 1)), k1))

            imag1 = np.r_[new1, imag]

            imag1 = np.r_[imag1, new1]

            imag1 = np.c_[new2, imag1]

            imag1 = np.c_[imag1, new2]

            y = np.zeros((a, b))

            sum2 = sum(sum(H))

            for i in range(k1, (k1 + a)):

                for j in range(k1, (k1 + b)):


                    y[(i - k1), (j - k1)] = relate(imag1[(i - k1):(i + k1 + 1), (j - k1):(j + k1 + 1)], H, k) / sum2


            return y



        #读取图像并滤波



        img =self.gray

        fltr=np.array(img, dtype='uint8')



        self.fltr=fltr
        print("滤波完成")
        plt.imshow(fltr)
        cv2.imwrite('fltr.jpg', fltr)

        return fltr


    def sobel(self) :


      #读取滤波后图像

      image=self.fltr


      width,height=self.width,self.height

      print(width,height)

      # 定义中间值
      sobel_cen=np. zeros((3,3))


      sobel_image=np.zeros((height,width))
      #用来存放生成的矩阵


      sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
      #sobel算子x轴方向


      sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
      #sobei算子y轴方向

      for x in range(height-2):


        for y in range(width-2):


         sobel_cen[0][0] = image[x][y]

         sobel_cen[0][1] = image[x][y + 1]

         sobel_cen[0][2] = image[x][y + 2]


         sobel_cen[1][0]= image[x+1][y]

         sobel_cen[1][1] = image[x + 1][y + 1]

         sobel_cen[1][2] = image[x + 1][y + 2]


         sobel_cen[2][0] = image[x + 2][y]

         sobel_cen[2][1] = image[x + 2][y + 1]

         sobel_cen[2][2] = image[x + 2][y + 2]


         Gx = np.sum(sobel_cen * sobel_x)

         Gy = np.sum(sobel_cen * sobel_y)


         sobel_image[x][y] = np.sqrt(Gx * Gx + Gy * Gy)


      result=np.trunc(sobel_image)

      result=result.astype(int)

      self.sobel_image=result

      print('检测完成')

      return result

    def encode(self):


        print(self.sobel_image)

        #储存图片

        cv2.imwrite('sobel.jpg', self.sobel_image)






"""main"""




test=edge('1.jpg')

test.read()


test.fliter()

test.sobel()

test.encode()


