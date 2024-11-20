import numpy as np
import cv2
import glob
import time

# thresh_size xác định kích thước của vùng sẽ đặt lại giá trị xung quanh mỗi điểm nhiễu để các tần số lân cận cũng bị loại bỏ.
def denoise_periodic(img, filter_size=2, center_threshold_factor=20):
    height, width = img.shape[:]
    img_fft = np.fft.fft2(img)
    shifted_fft = np.fft.fftshift(img_fft)

    # Hiển thị biến đổi Fourier bằng log của fft
    magnitude_spectrum = 20 * np.log(np.abs(shifted_fft))

    # Lấy giá trị tại điểm trung tâm
    center_value = magnitude_spectrum[int(height / 2)][int(width / 2)]

    # Tìm điểm có tần số cao
    # Tạo bộ lọc nhân chập có giá trị trung tâm là -1 và xung quanh là giá trị dương. 
    # Bộ lọc này giúp phát hiện các điểm tần số nhiễu cao bằng cách làm nổi bật các điểm có tần số cao so với tần số trung bình xung quanh.
    freq_magnitude = np.copy(magnitude_spectrum)
    # tạo ma trận có độ lớn [2 * filter_size + 1, 2 * filter_size + 1], điểm trung tâm set = 1
    convolution_kernel = np.ones((2 * filter_size + 1, 2 * filter_size + 1), np.float32) / ((2 * filter_size + 1) * (2 * filter_size + 1) - 1)
    convolution_kernel[filter_size][filter_size] = -1
    convolution_kernel = -convolution_kernel
    
    filtered_freq = cv2.filter2D(freq_magnitude, -1, convolution_kernel)

    center_threshold = center_value * center_threshold_factor / 356


    filtered_freq[int(height / 2)][int(width / 2)] = 0
    high_freq_indices = np.where(filtered_freq > center_threshold)

    # Loại bỏ các điểm không phải là cực đại
    # tọa độ của các điểm tần số cao được chọn lọc
    selected_x = []
    selected_y = []
    # print ("filtered_freq", filtered_freq)
    # print(f"Maximum value in filtered_freq: {np.max(filtered_freq)}")
    # print(f"Minimum value in filtered_freq: {np.min(filtered_freq)}")
    # print ("center_threshold", center_threshold)

    # print ("high_freq_indices", high_freq_indices)

    for i, _ in enumerate(high_freq_indices[0]):
        point_value = magnitude_spectrum[high_freq_indices[0][i]][high_freq_indices[1][i]]
        local_area = np.copy(magnitude_spectrum[
            max(0, high_freq_indices[0][i] - filter_size):min(height, high_freq_indices[0][i] + filter_size + 1),
            max(0, high_freq_indices[1][i] - filter_size):min(width, high_freq_indices[1][i] + filter_size + 1)
        ])
        local_area[filter_size][filter_size] = 0

        max_local_value = np.amax(local_area)
        # print("chajy vao day")


        if point_value - max_local_value < 20:
            continue
        selected_x.append(high_freq_indices[0][i])
        selected_y.append(high_freq_indices[1][i])
    # print("selected_x, selected_y", selected_x, selected_y)

    # Đặt giá trị tại các điểm nhiễu đã chọn về 1
    for i, _ in enumerate(selected_x):
        x =  selected_x[i] 
        y =  selected_y[i]
        shifted_fft[x, y] = 1

    # Khôi phục ảnh sau khi khử nhiễu
    inverse_shifted_fft = np.fft.ifftshift(shifted_fft)
    img_denoised = np.fft.ifft2(inverse_shifted_fft)
    img_denoised = np.abs(img_denoised).astype(np.uint8)

    return img_denoised
def count_objects(img):
    # cv2.imshow("img",img)

    # Khử nhiễu chung
    median_filtered = cv2.medianBlur(img, 5)
    # print("chajy vao day")
    # cv2.imshow("median_filtered",median_filtered)
    gaussian_blurred = cv2.GaussianBlur(median_filtered, (5, 5), 0)
    
    # Mở rộng ảnh để các bước lọc không ảnh hưởng đến các vùng ở cạnh
    # lật ảnh 3 kiểu
    flipped_all = cv2.flip(gaussian_blurred, -1)
    # cv2.imshow("flipped_all", flipped_all)
    
    flipped_vertical = cv2.flip(gaussian_blurred, 0)
    flipped_horizontal = cv2.flip(gaussian_blurred, 1)

    # gắn ảnh đã lật thành 3*3 ảnh
    row_top = np.concatenate((flipped_all, flipped_vertical, flipped_all), axis=1)
    row_middle = np.concatenate((flipped_horizontal, gaussian_blurred, flipped_horizontal), axis=1)
    row_bottom = np.concatenate((flipped_all, flipped_vertical, flipped_all), axis=1)

    expanded_img = np.concatenate((row_top, row_middle, row_bottom), axis=0)
    # cv2.imshow("expanded_img", expanded_img)

    # Xác định lại kích thước sau khi mở rộng
    expanded_height, expanded_width = expanded_img.shape[:]
    x1, x2 = int(expanded_width / 3), int(2 * expanded_width / 3)
    y1, y2 = int(expanded_height / 3), int(2 * expanded_height / 3)  

    # Cân bằng histogram cục bộ
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(int(expanded_width / 50), int(expanded_height / 50)))
    equalized_img = clahe.apply(expanded_img)
    # cv2.imshow("equalized_img_2", equalized_img)

    # làm mịn ảnh
    equalized_img = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    # cv2.imshow("equalized_img_3", equalized_img)


    # Ngưỡng hóa cục bộ
    # Nhị phân ảnh
    thresholded_img = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, -12)
    cropped_thresholded_img = thresholded_img[y1:y2, x1:x2]
    # cv2.imshow("cropped_thresholded_img", cropped_thresholded_img)
    equalized_img = equalized_img[y1:y2, x1:x2]


    # Tạo kernel để làm giãn và xói ảnh, giúp loại nhiễu và tách các đối tượng gần nhau
    # small_kernel erosion với kernel nhỏ làm giảm kích thước các đối tượng nhỏ, loại bỏ các điểm nhiễu nhỏ xung quanh.
    # small_kernel dilation sau đó giúp phục hồi các đối tượng chính trong ảnh, đảm bảo các chi tiết quan trọng không bị mất đi.
    # large_kernel Loại bỏ các kết nối mỏng hoặc vùng không mong muốn giữa các đối tượng lớn.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

    # Giãn và xói ảnh để loại bỏ nhiễu
    processed_img = cv2.erode(cropped_thresholded_img, small_kernel, iterations=1)
    processed_img = cv2.dilate(processed_img, small_kernel, iterations=1)
    processed_img = cv2.erode(processed_img, large_kernel, iterations=1)

    # Tìm các đường viền (contours)
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm đường viền lớn nhất 
    count = 0
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        max_area = max(area, max_area)

    # Đếm các đối tượng có diện tích đủ lớn so với đối tượng lớn nhất
    for contour in contours:
        # giảm bớt số điểm trong đường viền, giúp đơn giản hóa hình dạng của đối tượng.
        approx_contour = cv2.approxPolyDP(contour, 3, True)
        bounding_rect = cv2.boundingRect(approx_contour)

        area = cv2.contourArea(contour)
        if area < 0.08 * max_area:
            continue

        count += 1
        # ve hình vuong
        cv2.rectangle(equalized_img, (int(bounding_rect[0]), int(bounding_rect[1])),
                      (int(bounding_rect[0] + bounding_rect[2]), int(bounding_rect[1] + bounding_rect[3])), (0, 255, 0), 1)
        # cv2.imshow("img2", equalized_img)
        # ve duong vien
        cv2.drawContours(equalized_img, contours, -1, (0, 0, 255), 1)
        # cv2.imshow("img3", equalized_img)


    return equalized_img, count

i=0
# cv2.namedWindow("rice count", cv2.WINDOW_AUTOSIZE)
for path in glob.glob("images/*"):
    # print("image file: ", path)
    t = time.time()
    img_ori = cv2.imread(path)

    img_gray = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
    h,w = img_gray.shape[:]
    # print("h,w", h,w, img_gray)
    # cv2.imshow("img_gray ", img_gray )

    img_denoised = denoise_periodic(img_gray)
    # cv2.imshow("img_denoised ", img_denoised)
    cv2.waitKey(0)
    img_denoise = img_denoised.copy()
    # cv2.imshow("img_denoised ", img_denoised)


    res_img, res_count = count_objects(img_denoised)
    cv2.imshow("res_img", res_img)
    # print("time: ", time.time()-t)
    # print("rice count of "+ path, res_count)
    cv2.imshow("rice count of "+ path +": " + str(res_count), res_img)
    # cv2.moveWindow("rice count of "+ path +": " + str(res_count), 460*i,0)
    i = i +1

cv2.waitKey()
