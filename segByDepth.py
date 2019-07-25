import cv2
import numpy as np
from PIL import Image


# Run only if this module is being run directly
if __name__ == '__main__':
    depth_cv = cv2.imread("/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/PyTorch-ENet-master/icome/icome_test_images/00000-depth.png")
    img_cv = cv2.imread("/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/PyTorch-ENet-master/icome/icome_test_images/00000-color.png")
    kernel = np.ones((11,11), np.uint8)

    img_filter = cv2.GaussianBlur(img_cv, (11, 11), 0)
    img_filter = cv2.GaussianBlur(img_filter, (7, 7), 0)
    img_filter = cv2.GaussianBlur(img_filter, (5, 5), 0)
    img_filter = cv2.GaussianBlur(img_filter, (3, 3), 0)

    
    ########## plan 2 #############
    img_filter_np = np.asarray(img_filter).astype(np.float32)
    img_cv_np = np.asarray(img_cv).astype(np.float32)
    pil_open_np = np.asarray(depth_cv)
    print (np.max(pil_open_np))
   
    max_gray = np.max(pil_open_np)
    idx = pil_open_np < 1
    pil_open_np[idx] = max_gray

    min_gray = np.min(pil_open_np)
    print (max_gray, min_gray)

    focus = min_gray
    pos = np.where((pil_open_np > focus - 20) & (pil_open_np < focus + 20))
    print (pos)
    pil_open_np[pos] = focus

    img_filter2 = img_filter_np * (pil_open_np - min_gray) / (max_gray - min_gray)  + img_cv_np * (max_gray - pil_open_np) / (max_gray - min_gray)
    # img_filter2 = img_filter_np * (1 - np.exp(-1 * abs(pil_open_np - focus)))  + img_cv_np * np.exp(-1 * abs(pil_open_np - focus))

    img_out = Image.fromarray(cv2.cvtColor(img_filter2.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    img_out.save('/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/PyTorch-ENet-master/icome/icome_test_images/' + 'test.png')
    # Image.fromarray(cv2.cvtColor(pil_open.astype(np.uint8), cv2.COLOR_BGR2RGB)).save('/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/PyTorch-ENet-master/icome/icome_test_images/' + str(count + 500) + '.png')


