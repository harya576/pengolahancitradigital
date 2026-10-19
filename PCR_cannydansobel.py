import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('daun.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Gambar 'daun.jpg' tidak ditemukan di folder yang sama.")
else:
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    canny_edges = cv2.Canny(img_blur, 100, 200)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_blur, cmap='gray')
    plt.title('Grayscale + Gaussian Blur')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Deteksi Tepi: Sobel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Deteksi Tepi: Canny')
    plt.axis('off')

    plt.tight_layout()
    plt.show()