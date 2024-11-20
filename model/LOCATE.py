import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
from data.AG20K import OBJECT_AFFORDANCE_LIST

class LOCATE(data.Dataset):
    def __init__(self, image_root, crop_size=224,):
        self.image_root = image_root
        self.image_list = []
        self.crop_size = crop_size
        self.aff_list = ['beat', "boxing", "brush_with", "carry", "catch",
                            "cut", "cut_with", "drag", 'drink_with', "eat",
                            "hit", "hold", "jump", "kick", "lie_on", "lift",
                            "look_out", "open", "pack", "peel", "pick_up",
                            "pour", "push", "ride", "sip", "sit_on", "stick",
                            "stir", "swing", "take_photo", "talk_on", "text_on",
                            "throw", "type_on", "wash", "write"] 
        self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                            'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                            'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                            'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                            'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                            'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                            'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                            'tennis_racket', 'toothbrush', 'wine_glass']

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])

        files = os.listdir(self.image_root)
        for file in files:
            file_path = os.path.join(image_root, file)
            images = os.listdir(file_path)
            for img in images:
                    img_path = os.path.join(file_path, img)
                    self.image_list.append(img_path)

    def get_obj2aff_dict(self):
        OBJECT_TO_AFFORDANCE_LIST = {}
        for affordance, objects in OBJECT_AFFORDANCE_LIST.items():
            for obj in objects:
                if obj not in OBJECT_TO_AFFORDANCE_LIST:
                    OBJECT_TO_AFFORDANCE_LIST[obj] = []
                OBJECT_TO_AFFORDANCE_LIST[obj].append(affordance)
        return OBJECT_TO_AFFORDANCE_LIST
    
    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img_w = img.size[0]
        img_h = img.size[1]
        img = self.transform(img)
        return img, img_w, img_h

    
    def __getitem__(self, item):
        image_path = self.image_list[item]
        names = image_path.split("/")
        object = names[-2]
        object_aff_list = self.get_obj2aff_dict()[object]
        labels = []
        for aff in object_aff_list:
            labels.append(self.aff_list.index(aff))

        image, img_w, img_h = self.load_img(image_path)

        return image, labels, object, img_w, img_h

    def __len__(self):
        return len(self.image_list)
