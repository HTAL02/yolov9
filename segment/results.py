from utils.segment.general import masks2segments
from utils.general import scale_segments
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from copy import deepcopy
import cv2
import numpy as np
import torch

class BasicResults():
    def __init__(self, data, original_shape=None):
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.original_shape = original_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.original_shape)

    def __iter__(self):
        for box in self.data:
            yield box

    def __str__(self):
        return str(self.data)
    
    @property
    def shape(self):
        return self.data.shape

class Boxes(BasicResults):
    def __init__(self, boxes, original_shape=None):
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)
        # self.boxes = boxes
        super().__init__(boxes, original_shape)
        self.original_shape = original_shape
    
    @property
    def xyxy(self):
        return self.data[:, :4]
    
    @property
    def conf(self):
        return self.data[:, 4:5]
    
    @property
    def cls(self):
        return self.data[:, 5:6]
    
class Masks(BasicResults):
    def __init__(self, masks, original_shape=None):
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)
        # self.masks = masks
        super().__init__(masks, original_shape)
        self.original_shape = original_shape

    def int(self):
        """
        Convert the mask to int.
        """
        return self.data.int()
    
    @property
    def xy(self):
        segments = reversed(masks2segments(self.data))
        return [scale_segments(self.data.shape[1:], x, self.original_shape, normalize=False) for x in segments]
    
    @property
    def xyn(self):
        segments = reversed(masks2segments(self.data))
        return [scale_segments(self.data.shape[1:], x, self.original_shape, normalize=True) for x in segments]
    

class Results():
    def __init__(self, original_img, names, boxes=None, masks=None, original_shape=None):
        self.original_img = original_img
        if boxes is not None :
            self.boxes = Boxes(boxes, original_shape)
        if masks is not None :
            self.masks = Masks(masks, original_shape)
        self.original_shape = original_shape
        self.names = names
        self._keys = "boxes", "masks"
    
    def __len__(self):
        """
        Return the number of boxes.
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None :
                return(len(v))
    
    def __getitem__(self, idx):
        return self._apply("__getitem__", idx)
    
    def __str__(self):
        return f'{self.__class__.__name__}(boxes={self.boxes}, masks={self.masks})'
    def __repr__(self):
        return f'{self.__class__.__name__}(boxes={self.boxes}, masks={self.masks})'
    
    def _apply(self, func, *args, **kwargs):
        """
        Apply a function to the boxes and masks.
        Args:
            func (callable): The function to apply.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        returns:
            self: The modified Results object.
       
        """
        r = Results(original_img=self.original_img, names=self.names, original_shape=self.original_shape)
        for k in self._keys:
            if hasattr(self, k):
                v = getattr(self, k)
                if v is not None:
                    setattr(r, k, getattr(v, func)(*args, **kwargs))
        return r

    def plot(self, img=None, im_gpu= None, line_thickness=None, show=True, save_path=None, **kwargs):
        """
        Plot the results on the image.
        Args:
            img (numpy.ndarray): The image to plot on.
            show (bool): Whether to show the image.
            save_path (str): The path to save the image.
            **kwargs: Additional arguments for plotting.
        """
        # if  not self.boxes and not self.masks:
        #     return self.original_img if img is None else img
        # check mask and boxes are atrributes
        if hasattr(self, 'boxes') and  hasattr(self, 'masks'):
        
            annotator = Annotator(
                deepcopy(self.original_img if img is None else img), 
                line_width=line_thickness, 
                example=str(self.names))
            
            # Plot Segment results
            annotator.masks(self.masks.data ,
                    colors=[colors(x, True) for x in self.boxes.data[:, 5]],
                    im_gpu=im_gpu)

            # Plot Detect results
            for i, (xyxy, conf, cls) in enumerate(zip(self.boxes.xyxy, self.boxes.conf, self.boxes.cls)):
                c = int(cls)
                label = f'{self.names[c]} {conf.item():.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
            img = annotator.result()
                
            if save_path:
                cv2.imwrite(save_path, img)
            
            if show:
                cv2.imshow('Image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return img
        else:
            return self.original_img if img is None else img
        
    def retrive_data(self, cls):
        """
        Retrieve data of a specific class.
        Args:
            cls (int or str): The class index or name.
        Returns:
            Results: A new Results object with the data for the specified class.
        """
        if isinstance(cls, str):
            cls = self.names.index(cls)
        elif isinstance(cls, int):
            cls = [cls]
        else:
            raise TypeError("cls must be int or str")
        
        r = Results(original_img=self.original_img, names=self.names, original_shape=self.original_shape)
        
        return r

