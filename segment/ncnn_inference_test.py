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
from utils.general import non_max_suppression
from utils.augmentations import letterbox

def test_inference(source):
    torch.manual_seed(0)
    # in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    in0 = cv2.imread(source)
    imgsz = (640, 640)
    print(f"Input image shape: {in0.shape}")
    # in0 = cv2.resize(in0, imgsz)
    in0 = letterbox(in0, imgsz, stride=32, auto=False)[0] 
    # in0 = cv2.cvtColor(in0, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    in0 = in0.transpose((2, 0, 1))[::-1]
    in0 = np.ascontiguousarray(in0)
    print(f"Input image shape: {in0.shape}")
    in0 = torch.from_numpy(in0.copy()).unsqueeze(0).float()  # Convert to tensor and add batch dimension
    out = []

    with ncnn.Net() as net:
        net.load_param("models/yolov9_models/ncnn_2/best_furniture_person_seg.ncnn.param")
        net.load_model("models/yolov9_models/ncnn_2/best_furniture_person_seg.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    start_time = dt.now()
    result = test_inference('data/input/test_images/test_about_to_fall1.png')
    end_time = dt.now()
    print(f"Inference time: {(end_time - start_time).total_seconds()} seconds")
    if isinstance(result, tuple):
        for i, res in enumerate(result):
            print(f"Output {i}: {res.shape}")
        pred = non_max_suppression(result[0], conf_thres=0.5, iou_thres=0.005, nm=32, max_det=1000)
        print(f"Processed {len(pred[0])} detections")
    else:
        print(f"Output: {result.shape}")
        pred = non_max_suppression(result, conf_thres=0.5, iou_thres=0.65)
        print(f"Processed {len(pred[0])} detections")

    # print(test_inference())
