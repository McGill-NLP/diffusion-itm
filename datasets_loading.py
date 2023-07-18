import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import PIL
import numpy as np
import torch
from torchvision import datasets
from glob import glob
from aro.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
import pandas as pd
import ast
from datasets import load_dataset

def get_dataset(dataset_name, root_dir, transform=None, resize=512, scoring_only=False, tokenizer=None, split='val', max_train_samples=None, hard_neg=False, targets=None, neg_img=False, mixed_neg=False, details=False):
    if dataset_name == 'winoground':
        return WinogroundDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    if dataset_name == 'mmbias':
        return BiasDataset(root_dir, resize=resize, transform=transform, targets=targets)
    if dataset_name == 'genderbias':
        return GenderBiasDataset(root_dir, resize=resize, transform=transform)
    elif dataset_name == 'imagecode':
        return ImageCoDeDataset(root_dir, split, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'imagecode_video':
        return ImageCoDeDataset(root_dir, split, transform, resize=resize, scoring_only=scoring_only, static=False)
    elif dataset_name == 'flickr30k':
        return Flickr30KDataset(root_dir, transform, scoring_only=scoring_only, split=split, tokenizer=tokenizer, details=details)
    elif dataset_name == 'flickr30k_text':
        return Flickr30KTextRetrievalDataset(root_dir, transform, scoring_only=scoring_only, split=split, tokenizer=tokenizer, hard_neg=hard_neg, details=details)
    elif dataset_name == 'flickr30k_neg':
        return Flickr30KNegativesDataset(root_dir, transform, scoring_only=scoring_only, split=split, tokenizer=tokenizer, hard_neg=hard_neg)
    elif dataset_name == 'lora_flickr30k':
        return LoRaFlickr30KDataset(root_dir, transform, tokenizer=tokenizer, max_train_samples=max_train_samples)
    elif dataset_name == 'imagenet':
        return ImagenetDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'svo_verb':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, neg_type='verb')
    elif dataset_name == 'svo_subj':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, neg_type='subj')
    elif dataset_name == 'svo_obj':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, neg_type='obj')
    elif dataset_name == 'clevr':
        return CLEVRDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'pets':
        return PetsDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'vg_relation':
        return VG_Relation(image_preprocess=transform, download=True, root_dir=root_dir)
    elif dataset_name == 'vg_attribution':
        return VG_Attribution(image_preprocess=transform, download=True, root_dir=root_dir)
    elif dataset_name == 'coco_order':
        return COCO_Order(image_preprocess=transform, download=True, root_dir=root_dir)
    elif dataset_name == 'flickr30k_order':
        return Flickr30k_Order(image_preprocess=transform, download=True, root_dir=root_dir)
    elif dataset_name == 'mscoco':
        return MSCOCODataset(root_dir, transform, resize=resize, split=split, tokenizer=tokenizer, hard_neg=hard_neg, neg_img=neg_img, mixed_neg=mixed_neg)
    elif dataset_name == 'mscoco_val':
        return ValidMSCOCODataset(root_dir, transform, resize=resize, split='val', tokenizer=tokenizer, neg_img=neg_img, hard_neg=hard_neg)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

def diffusers_preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.squeeze(0)
    return 2.0 * image - 1.0
    

lora_train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512) if True else transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.data = datasets.ImageFolder(root_dir + '/val')
        # self.loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.resize = resize
        self.transform = transform
        self.classes = list(json.load(open(f'./imagenet_classes.json', 'r')).values())
        if True:
            prompted_classes = []
            for c in self.classes:
                class_text = 'a photo of a ' + c
                prompted_classes.append(class_text)
            self.classes = prompted_classes
        self.scoring_only = scoring_only

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = img.convert("RGB")
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        else:
            class_id = idx // 50

        if self.scoring_only:
            return self.classes, class_id
        else:
            return ([img], [img_resize]), self.classes, class_id

class PetsDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        
        self.root_dir = root_dir
        # read all imgs in root_dir with glob
        imgs = list(glob(root_dir + '/images/*.jpg'))
        self.resize = resize
        self.transform = transform
        self.classes = list(open(f'{root_dir}/classes.txt', 'r').read().splitlines())
        self.data = []
        for img_path in imgs:
            filename = img_path.split('/')[-1].split('_')
            class_name = ' '.join(filename[:-1])
            lower_case_class_name = class_name.lower()
            class_id = self.classes.index(lower_case_class_name)
            self.data.append((img_path, class_id))
        prompted_classes = []
        for c in self.classes:
            class_text = 'a photo of a ' + c
            prompted_classes.append(class_text)
        self.classes = prompted_classes
        self.scoring_only = scoring_only

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = Image.open(img)
            img = img.convert("RGB")
            if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
            else:
                img_resize = img.resize((self.resize, self.resize))
                img_resize = diffusers_preprocess(img_resize)
        else:
            class_id = idx // 50
        print(class_id)
        if self.scoring_only:
            return self.classes, class_id
        else:
            return [0, [img_resize]], self.classes, class_id

    def __len__(self):
        return len(self.data)

class GenderBiasDataset(Dataset):
    def __init__(self, root_dir, resize=512, transform=None, targets=None):
        self.root_dir = root_dir #datasets/genderbias/
        self.resize = resize
        self.transform = transform
        self.data = []
        # self.attributes = {'apron':'clothes','suit':'clothes','briefcase':'bags','purse':'bags','beer':'drinks','wine':'drinks'}
        clothes_imgs = list(glob(f'{root_dir}/suit-images/*.jpg')) + list(glob(f'{root_dir}/apron-images/*.jpg'))
        bags_imgs = list(glob(f'{root_dir}/briefcase-images/*.jpg')) + list(glob(f'{root_dir}/purse-images/*.jpg'))
        drinks_imgs = list(glob(f'{root_dir}/beer-images/*.jpg')) + list(glob(f'{root_dir}/wine-images/*.jpg'))
        self.data.extend((img,('clothes',['suit','apron'])) for img in clothes_imgs)
        self.data.extend((img,('bags',['briefcase','purse'])) for img in bags_imgs)
        self.data.extend((img,('drinks',['beer','wine'])) for img in drinks_imgs)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, (target, texts) = self.data[idx]
        # class_id is {male,female}_{clothes,drinks,bags}
        class_id = f'male_{target}' if int(img.split('/')[-1].split('.')[0]) <=6 else f'female_{target}'
        img = Image.open(img)
        img = img.convert("RGB")
        if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        return (0, [img_resize]), texts, class_id
    
class BiasDataset(Dataset):
    def __init__(self, root_dir, resize=512, transform=None, targets=None):
        self.root_dir = root_dir #datasets/mmbias/
        self.resize = resize
        self.transform = transform
        self.phrases = json.load(open(f'{root_dir}/textual_phrases.txt'))
        self.texts = self.phrases['pleasant_phrases']
        self.texts.extend(self.phrases['unpleasant_phrases']) # texts is ['good','texts',...,'bad','texts']
        self.classes = ['Christian','Muslim','Jewish','Hindu','American','Arab','Heterosexual','LGBT','Buddhist']
        if targets == '': # use all classes by default
            targets=self.classes
        print(targets)
        self.data = []
        
        # Religion
        if 'Christian' in targets:
            self.christian_imgs = list(glob(f'{root_dir}/data/Images/Religion/Christian/*.jpg'))
            self.data.extend([(img_path,0) for img_path in self.christian_imgs])
        if 'Muslim' in targets:
            self.muslim_imgs = list(glob(f'{root_dir}/data/Images/Religion/Muslim/*.jpg'))
            self.data.extend([(img_path,1) for img_path in self.muslim_imgs])
        if 'Jewish' in targets:
            self.jewish_imgs = list(glob(f'{root_dir}/data/Images/Religion/Jewish/*.jpg'))
            self.data.extend([(img_path,2) for img_path in self.jewish_imgs])
        if 'Hindu' in targets:
            self.hindu_imgs = list(glob(f'{root_dir}/data/Images/Religion/Hindu/*.jpg'))
            self.data.extend([(img_path,3) for img_path in self.hindu_imgs])
        if 'Buddhist' in targets:
            self.buddhist_imgs = list(glob(f'{root_dir}/data/Images/Religion/Buddhist/*.jpg'))
            self.data.extend([(img_path,8) for img_path in self.buddhist_imgs])
        # Nationality
        if 'American' in targets:
            self.american_imgs = list(glob(f'{root_dir}/data/Images/Nationality/American/*.jpg'))
            self.data.extend([(img_path,4) for img_path in self.american_imgs])
        if 'Arab' in targets:
            self.arab_imgs = list(glob(f'{root_dir}/data/Images/Nationality/Arab/*.jpg'))
            self.data.extend([(img_path,5) for img_path in self.arab_imgs])
        # Sexuality
        if 'Heterosexual' in targets:
            self.hetero_imgs = list(glob(f'{root_dir}/data/Images/Sexual Orientation/Heterosexual/*.jpg'))
            self.data.extend([(img_path,6) for img_path in self.hetero_imgs])
        if 'LGBT' in targets:
            self.lgbt_imgs = list(glob(f'{root_dir}/data/Images/Sexual Orientation/LGBT/*.jpg'))
            self.data.extend([(img_path,7) for img_path in self.lgbt_imgs])
        # uncommment for just subset
        # self.data = self.data[::5]
        # self.texts = self.texts[::3]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, class_id = self.data[idx]
        img = Image.open(img)
        img = img.convert("RGB")
        if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        return (0, [img_resize]), self.texts, class_id


class WinogroundDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        self.examples = load_dataset('facebook/winoground', use_auth_token='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')
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.examples['test'][idx]
        cap0 = ex['caption_0']
        cap1 = ex['caption_1']
        img_id = ex['id']
        if not self.scoring_only:
            img0 = ex['image_0']
            img1 = ex['image_1']
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
        text = [cap0, cap1]
        if self.scoring_only:
            return text, img_id
        else:
            return (0, [img0_resize, img1_resize]), text, img_id

class ImageCoDeDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512, scoring_only=False, static=True):
        self.root_dir = 'data/imagecode'
        self.resize = resize
        self.dataset = self.load_data(self.root_dir, split, static_only=static)
        self.transform = transform
        self.scoring_only = scoring_only

    @staticmethod
    def load_data(data_dir, split, static_only=True):
        split = 'valid' if split == 'val' else split
        with open(f'{data_dir}/{split}_data.json') as f:
            json_file = json.load(f)
        img_path = f'{data_dir}/image-sets'

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if static_only:
                    if static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))

        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.dataset[idx]
        if not self.scoring_only:
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_files]
            if self.transform:
                imgs_resize = [self.transform(img).unsqueeze(0) for img in imgs]
            else:
                imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
                imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        if self.scoring_only:
            return text, img_dir, img_idx
        else:
            return (0, imgs_resize), [text], img_dir, img_idx



class MSCOCODataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=True, neg_img=False, mixed_neg=False, tsv_path='aro/temp_data/train_neg_clip.tsv'):
        self.root_dir = 'data/mscoco/train2014'
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.all_texts = self.data['title'].tolist()
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        self.mixed_neg = mixed_neg
        self.rand_neg = not self.hard_neg and not self.neg_img


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = img_path.split('/')[-1]
        if 'train2014' in img_path:
            img_path = f"{self.root_dir}/{img_path}"
        else:
            img_path = f"data/coco_order/val2014/{img_path}"
        text = row['title']
        neg_captions =  ast.literal_eval(row['neg_caption'])
        neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]

        neg_img_ids = ast.literal_eval(row['neg_image']) # a list of row indices in self.data
        neg_paths = self.data.iloc[neg_img_ids]['filepath'].tolist()
        new_neg_paths = []
        for path in neg_paths:
            path = path.split('/')[-1]
            if 'train2014' in path:
                path = f"{self.root_dir}/{path}"
            else:
                path = f"data/coco_order/val2014/{path}"
            new_neg_paths.append(path)
        neg_paths = new_neg_paths
        
        
        if self.tokenizer:
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text0 = text.input_ids.squeeze(0)
            # text0 = text[0]
            if self.mixed_neg:
                text_neg = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_neg = text_neg.input_ids.squeeze(0)
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_neg, text_rand])
            elif self.hard_neg:
                text_rand = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            elif self.rand_neg:
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            else:
                text = text0
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img or self.mixed_neg:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        # if np.random.rand() > 0.99:
        #     print("Img true:", img_path)
        #     print("Neg Img:", rand_path)
        #     print(text)
        
        return [0, imgs], text, 0


class ValidMSCOCODataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=False, tsv_path='aro/temp_data/valid_neg_clip.tsv', neg_img=False):
        self.root_dir = 'data/mscoco/'
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        if not self.neg_img:
            self.hard_neg = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = img_path.split('/')[-1]
        img_path = f"data/coco_order/val2014/{img_path}"
        text = row['title']
        if self.hard_neg:
            neg_captions =  ast.literal_eval(row['neg_caption'])
            neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]
            text = [text, neg_caption]
        else:
            text = [text]

        neg_img_ids = ast.literal_eval(row['neg_image'])
        neg_paths = self.data.iloc[neg_img_ids]['filepath'].tolist()
        new_neg_paths = []
        for path in neg_paths:
            path = path.split('/')[-1]
            if 'train2014' in path:
                path = f"{self.root_dir}/{path}"
            else:
                path = f"data/coco_order/val2014/{path}"
            new_neg_paths.append(path)
        neg_paths = new_neg_paths

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        # print("Img true:", img_path)
        # print("Neg Img:", rand_path)
        # print(text)

        return [0, imgs], text, 0


class Flickr30KDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split='val', tokenizer=None, first_query=True, details=False):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{root_dir}/{split}_top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        # get only every 5th example
        if first_query:
            self.data = self.data[::5]
        self.transform = transform
        self.scoring_only = scoring_only
        self.tokenizer = tokenizer
        self.details = details
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        if self.tokenizer:
            text = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
        img_paths = ex[1]
        img_idx = 0
        if not self.scoring_only:
            imgs = [Image.open(f'{img_path}').convert("RGB") for img_path in img_paths]
            imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
            #convert pillow to numpy array
            # imgs_resize = [np.array(img) for img in imgs_resize]
            imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            else:
                imgs = [transforms.ToTensor()(img) for img in imgs]
        if self.scoring_only:
            return [text], img_idx
        else:
            return [img_paths, imgs_resize], [text], img_idx

class Flickr30KTextRetrievalDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split='val', tokenizer=None, hard_neg=False, details=False):
        self.root_dir = 'data/flickr30k'
        self.resize = resize
        self.data = json.load(open(f'{self.root_dir}/{split}_top10_RN50x64_text.json', 'r'))
        if split == 'val':
            self.data = list(self.data.items()) # dictionary from img_path to list of 10 captions
        self.all_captions = []
        for img_path, captions in self.data:
            self.all_captions.extend(captions)
        self.transform = transform
        self.scoring_only = scoring_only
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.details = details

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img_path = ex[0]
        text = ex[1]
        if self.tokenizer:
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
            text0 = text[0]
            if self.hard_neg:
                text_rand = text[np.random.randint(5, len(text))]
            else:
                # get text from self.all_captions
                text_rand = self.all_captions[np.random.randint(0, len(self.all_captions))]
                text_rand = self.tokenizer(text_rand, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
            text = torch.stack([text0, text_rand])
        img = Image.open(f'{img_path}').convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)

        return [img_path, [img_resize]], text, 0

class Flickr30KNegativesDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split='val', tokenizer=None, hard_neg=False):
        self.root_dir = 'data/flickr30k'
        self.resize = resize
        self.data = json.load(open(f'{self.root_dir}/{split}_top10_RN50x64_text.json', 'r'))
        if split == 'val':
            self.data = list(self.data.items()) # dictionary from img_path to list of 10 captions
        self.all_captions = []
        for img_path, captions in self.data:
            self.all_captions.extend(captions)

        self.txt2img = json.load(open(f'{self.root_dir}/{split}_top10_RN50x64.json', 'r'))
        self.transform = transform
        self.scoring_only = scoring_only
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img_path = ex[0]
        strings = ex[1]
        if self.tokenizer:
            text = self.tokenizer(strings, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
            text0 = text[0]
            if self.hard_neg:
                rand_idx = np.random.randint(5, len(text))
                text_rand = text[rand_idx]
                string_rand = strings[rand_idx]
                img_rand = self.txt2img[string_rand][0]
            else:
                # get text from self.all_captions
                text_rand = self.all_captions[np.random.randint(0, len(self.all_captions))]
                img_rand = self.txt2img[text_rand][0]
                text_rand = self.tokenizer(text_rand, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
            img_rand = Image.open(f'{img_rand}').convert("RGB")
            img_rand_resize = img_rand.resize((self.resize, self.resize))
            img_rand_resize = diffusers_preprocess(img_rand_resize)
            empty_text = ''
            empty_text = self.tokenizer(empty_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            empty_text = empty_text.input_ids.squeeze(0)
            text = torch.stack([text0, text_rand, empty_text])
        img = Image.open(f'{img_path}').convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)

            return [0, [img_resize, img_rand_resize]], text, 0
 

class LoRaFlickr30KDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, tokenizer=None, max_train_samples=None):
        self.root_dir = root_dir
        self.resize = resize
        self.max_train_samples = max_train_samples
        self.data = json.load(open(f'{root_dir}/train_top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        if self.max_train_samples is not None:
            self.data = self.data[:self.max_train_samples]
        self.transform = transform
        self.tokenizer = tokenizer
        self.two_imgs = True
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        img_paths = ex[1]
        img_idx = 0
        if self.two_imgs:
            text = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
            img0 = Image.open(img_paths[0]).convert("RGB")
            img_rand = Image.open(random.choice(img_paths[1:])).convert("RGB")
            imgs = [img0, img_rand]
        else:
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
            text = [text]
        #convert pillow to numpy array
        # imgs_resize = [np.array(img) for img in imgs]
        imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        return imgs_resize, text, img_idx

class SVOClassificationDataset(Dataset):

    def __init__(self, root_dir, transform, resize=512, scoring_only=False, neg_type='verb'):
        self.transform = transform
        self.root_dir = 'data/svo'
        self.data = self.load_data(self.root_dir, neg_type=neg_type)
        self.resize = resize
        self.scoring_only = scoring_only

    def load_data(self, data_dir, neg_type='verb'):
        dataset = []
        split_file = os.path.join(data_dir, 'svo.json')
        with open(split_file) as f:
            json_file = json.load(f)

        for i, row in enumerate(json_file):
            if row['neg_type'] != neg_type:
                continue
            pos_id = str(row['pos_id'])
            neg_id = str(row['neg_id'])
            sentence = row['sentence']
            # get two different images
            pos_file = os.path.join(data_dir, "images", pos_id)
            neg_file = os.path.join(data_dir, "images", neg_id)
            dataset.append((pos_file, neg_file, sentence))

        return dataset
    
    def __getitem__(self, idx):
        file0, file1, text = self.data[idx]
        img0 = Image.open(file0).convert("RGB")
        img1 = Image.open(file1).convert("RGB")
        if not self.scoring_only:
            imgs = [img0, img1]
            if self.transform:
                imgs_resize = [self.transform(img).unsqueeze(0) for img in imgs]
            else:
                imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
                imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        if self.scoring_only:
            return [text], 0
        else:
            return (0, imgs_resize), [text], 0
 
        
    def __len__(self):
        return len(self.data)

class CLEVRDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        root_dir = '../clevr/validation'
        self.root_dir = root_dir
        subtasks = ['pair_binding_size', 'pair_binding_color', 'recognition_color', 'recognition_shape', 'spatial', 'binding_color_shape', 'binding_shape_color']
        data_ = []
        for subtask in subtasks:
            self.data = json.load(open(f'{root_dir}/captions/{subtask}.json', 'r')).items()
            for k, v in self.data:
                for i in range(len(v)):
                    if 'subtask' == 'spatial':
                        texts = [v[i][1], v[i][2]]
                    else:
                        texts = [v[i][0], v[i][1]]
                    data_.append((k, texts, subtask))
        self.data = data_
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        cap0 = ex[1][0]
        cap1 = ex[1][1]
        img_id = ex[0]
        subtask = ex[2]
        img_path0 = f'{self.root_dir}/images/{img_id}'
        if not self.scoring_only:
            img0 = Image.open(img_path0).convert("RGB")
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)

        text = [cap0, cap1]
        if self.scoring_only:
            return text, 0
        else:
            return (0, [img0_resize]), text, subtask, 0
