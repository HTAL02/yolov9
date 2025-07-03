import numpy as np
import ncnn, cv2
import torch
from datetime import datetime as dt
from pathlib import Path
import sys, os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.general import non_max_suppression, scale_boxes
from utils.segment.general import masks2segments, process_mask
from utils.augmentations import letterbox
from utils.general import scale_boxes
from results import Masks

def test_inference(frame):
    torch.manual_seed(0)
    # in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    imgsz = (640, 640)
    print(f"Input image shape: {frame.shape}")
    in0 = frame.copy()
    # in0 = cv2.resize(in0, imgsz)
    scaled_img = letterbox(in0, imgsz, stride=32, auto=False)[0] 
    # in0 = cv2.cvtColor(in0, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # in0 = in0.transpose((2, 0, 1))[::-1]
    # in0 = np.ascontiguousarray(in0)
    # print(f"Input image shape: {in0.shape}")
    # in0 = torch.from_numpy(in0.copy()).unsqueeze(0).float()  # Convert to tensor and add batch dimension
    w, h = scaled_img.shape[:2]
    print(f"Scaled image shape: {scaled_img.shape}")
    image = ncnn.Mat.from_pixels(np.asarray(scaled_img), ncnn.Mat.PixelType.PIXEL_BGR, w, h)
    mean = [0,0,0]
    std = [1/255,1/255,1/255]
    # Normalize the image
    image.substract_mean_normalize(mean=mean, norm=std)
    out = []

    with ncnn.Net() as net:
        net.load_param("models/yolov9_models/ncnn_5/best_furniture_person_seg.ncnn.param")
        net.load_model("models/yolov9_models/ncnn_5/best_furniture_person_seg.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", image)

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":

    frame = cv2.imread('data/input/test_images/test_about_to_fall1.png')

    start_time = dt.now()
    result = test_inference(frame)
    end_time = dt.now()
    print(f"Inference time: {(end_time - start_time).total_seconds()} seconds")
    if isinstance(result, tuple):
        im = cv2.resize(frame, (640, 640))

        for i, res in enumerate(result):
            print(f"Output {i}: {res.shape}")
        pred = non_max_suppression(result[0], conf_thres=0.5, iou_thres=0.7, nm=32, max_det=1000)
        det = pred[0]
        proto = result[1]
        if len(proto) == 3:
            proto = proto[2]
            
        masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[:2], upsample=True)  # HWC
        det[:, :4] = scale_boxes(im.shape[:2], det[:, :4], frame.shape).round()  # rescale boxes to im0 size
        
        print(f'mask shape : {masks.shape}')
        # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()  # rescale boxes to im0 size
        print(f"Processed {len(det)} detections, shape: {det.shape}")
        print(f"Detection results: {det[:, :6]}")

        mask = Masks(masks, frame.shape[:2])
        for i, res in enumerate(det):
            x, y, w, h, conf, cls = res[:6]
            x1, y1, x2, y2 = x , y  , w , h
            
            # draw rectangle on the image
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{int(cls)}: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # draw the mask with fillpolygon
            m = mask[i].xy

            cv2.fillPoly(frame, np.int32([m]), (0, 0, 255))
        # save the image with detections
        cv2.imwrite('data/detected_image_0.jpg', frame)
    else:
        print(f"Output: {result.shape}")
        pred = non_max_suppression(result, conf_thres=0.5, iou_thres=0.65)
        det = pred[0]
        print(f"Processed : \n{len(det)} detections, shape: {det.shape}")

    # print(test_inference())
