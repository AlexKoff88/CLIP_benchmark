import os
from collections import defaultdict
from torchvision.datasets import (
    VisionDataset,
    CIFAR10, CIFAR100, ImageNet, CocoCaptions, Flickr8k, Flickr30k, Food101, SUN397,
    StanfordCars, FGVCAircraft, DTD, OxfordIIITPet, Caltech101, Flowers102,
    MNIST, STL10, EuroSAT, GTSRB, Kitti, Country211, PCAM, RenderedSST2
)
from . import voc2007, flickr
from torch.utils.data import default_collate


def build_dataset(args, transform, train=False, download=True, **kwargs):
    dataset_name = args.dataset
    root = args.dataset_root
    if dataset_name == "cifar10":
        return CIFAR10(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "cifar100":
        return CIFAR100(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "imagenet1k":
        return ImageNet(root=root, split="train" if train else "val", transform=transform, **kwargs)
    elif dataset_name == "voc2007":
        return voc2007.PASCALVoc2007(root=root, set="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "mscoco_captions":
        return CocoCaptions(root=root, annFile=args.annotation_file, transform=transform, **kwargs)
    elif dataset_name == "flickr30k":
        return flickr.Flickr(root=root, ann_file=args.annotation_file, transform=transform, **kwargs)
    elif dataset_name == "flickr8k":
        return flickr.Flickr(root=root, ann_file=args.annotation_file, transform=transform, **kwargs)
    elif dataset_name == "food101":
        ds = Food101(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = list(map(lambda s:" ".join(s.split("_")), ds.classes))
        return ds
    elif dataset_name == "sun397":
        ds = SUN397(root=root, transform=transform, download=download, **kwargs)
        ds.classes = list(map(lambda s:" ".join(s.split("_")), ds.classes ))
        ds.classes = list(map(lambda s:" ".join(s.split("/")), ds.classes ))
        return ds
    elif dataset_name == "cars":
        return StanfordCars(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "fgvc_aircraft":
        return FGVCAircraft(root=root, annotation_level="variant", split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "dtd":
        return DTD(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "pets":
        return OxfordIIITPet(root=root, split="train" if train else "test", target_types="category", transform=transform, download=download, **kwargs)
    elif dataset_name == "caltech101":
        # broken download link (can't download google drive), fixed by this PR https://github.com/pytorch/vision/pull/5645
        return Caltech101(root=root, target_type="category", transform=transform, download=download, **kwargs)
    elif dataset_name == "flowers":
        ds = Flowers102(root=root, split="train" if train else "test", transform=transform, target_transform=lambda y:y-1, download=download, **kwargs)
        ds.classes = flowers_classes
        return ds
    elif dataset_name == "mnist":
        ds = MNIST(root=root, train=train, transform=transform, download=download, **kwargs)
        ds.classes = list(map(str, range(10)))
        return ds
    elif dataset_name == "stl10":
        return STL10(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "eurosat":
        ds = EuroSAT(root=root, transform=transform, download=download, **kwargs)
        ds.classes = eurosat_classes
        return ds
    elif dataset_name == "gtsrb":
        ds =  GTSRB(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = gtsrb_classes
        return ds
    elif dataset_name == "kitti":
        # no classnames, needs to target_transform
        ds = Kitti(root=root, train=train, transform=transform, download=download, **kwargs)
        ds.classes = kitti_classes
        return ds
    elif dataset_name == "country211":
        ds = Country211(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = country211_classes
        return ds
    elif dataset_name == "pcam":
        #  google drive links are not downloadead, fixed by this PR https://github.com/pytorch/vision/pull/5645
        return PCAM(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "renderedsst2":
        return RenderedSST2(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

def get_dataset_collate_fn(dataset_name):
    if dataset_name in ("mscoco_captions", "flickr30k", "flickr8k"):
        return image_captions_collate_fn
    else:
        return default_collate

def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts

zeroshot_classification_templates = {
    "cifar10": [
        "a photo of a {c}.",
        "a blurry photo of a {c}.",
        "a black and white photo of a {c}.",
        "a low contrast photo of a {c}.",
        "a high contrast photo of a {c}.",
        "a bad photo of a {c}.",
        "a good photo of a {c}.",
        "a photo of a small {c}.",
        "a photo of a big {c}.",
        "a photo of the {c}.",
        "a blurry photo of the {c}.",
        "a black and white photo of the {c}.",
        "a low contrast photo of the {c}.",
        "a high contrast photo of the {c}.",
        "a bad photo of the {c}.",
        "a good photo of the {c}.",
        "a photo of the small {c}.",
        "a photo of the big {c}."
    ],
    "cifar100":[
        "a photo of a {c}.",
        "a blurry photo of a {c}.",
        "a black and white photo of a {c}.",
        "a low contrast photo of a {c}.",
        "a high contrast photo of a {c}.",
        "a bad photo of a {c}.",
        "a good photo of a {c}.",
        "a photo of a small {c}.",
        "a photo of a big {c}.",
        "a photo of the {c}.",
        "a blurry photo of the {c}.",
        "a black and white photo of the {c}.",
        "a low contrast photo of the {c}.",
        "a high contrast photo of the {c}.",
        "a bad photo of the {c}.",
        "a good photo of the {c}.",
        "a photo of the small {c}.",
        "a photo of the big {c}."
    ],
    "imagenet1k": [
        "a bad photo of a {c}.",
        "a photo of many {c}.",
        "a sculpture of a {c}.",
        "a photo of the hard to see {c}.",
        "a low resolution photo of the {c}.",
        "a rendering of a {c}.",
        "graffiti of a {c}.",
        "a bad photo of the {c}.",
        "a cropped photo of the {c}.",
        "a tattoo of a {c}.",
        "the embroidered {c}.",
        "a photo of a hard to see {c}.",
        "a bright photo of a {c}.",
        "a photo of a clean {c}.",
        "a photo of a dirty {c}.",
        "a dark photo of the {c}.",
        "a drawing of a {c}.",
        "a photo of my {c}.",
        "the plastic {c}.",
        "a photo of the cool {c}.",
        "a close-up photo of a {c}.",
        "a black and white photo of the {c}.",
        "a painting of the {c}.",
        "a painting of a {c}.",
        "a pixelated photo of the {c}.",
        "a sculpture of the {c}.",
        "a bright photo of the {c}.",
        "a cropped photo of a {c}.",
        "a plastic {c}.",
        "a photo of the dirty {c}.",
        "a jpeg corrupted photo of a {c}.",
        "a blurry photo of the {c}.",
        "a photo of the {c}.",
        "a good photo of the {c}.",
        "a rendering of the {c}.",
        "a {c} in a video game.",
        "a photo of one {c}.",
        "a doodle of a {c}.",
        "a close-up photo of the {c}.",
        "a photo of a {c}.",
        "the origami {c}.",
        "the {c} in a video game.",
        "a sketch of a {c}.",
        "a doodle of the {c}.",
        "a origami {c}.",
        "a low resolution photo of a {c}.",
        "the toy {c}.",
        "a rendition of the {c}.",
        "a photo of the clean {c}.",
        "a photo of a large {c}.",
        "a rendition of a {c}.",
        "a photo of a nice {c}.",
        "a photo of a weird {c}.",
        "a blurry photo of a {c}.",
        "a cartoon {c}.",
        "art of a {c}.",
        "a sketch of the {c}.",
        "a embroidered {c}.",
        "a pixelated photo of a {c}.",
        "itap of the {c}.",
        "a jpeg corrupted photo of the {c}.",
        "a good photo of a {c}.",
        "a plushie {c}.",
        "a photo of the nice {c}.",
        "a photo of the small {c}.",
        "a photo of the weird {c}.",
        "the cartoon {c}.",
        "art of the {c}.",
        "a drawing of the {c}.",
        "a photo of the large {c}.",
        "a black and white photo of a {c}.",
        "the plushie {c}.",
        "a dark photo of a {c}.",
        "itap of a {c}.",
        "graffiti of the {c}.",
        "a toy {c}.",
        "itap of my {c}.",
        "a photo of a cool {c}.",
        "a photo of a small {c}.",
        "a tattoo of the {c}."
    ],
    "food101":[
        'a photo of {c}, a type of food.'
    ],
    "sun397":[
        'a photo of a {c}.',
        'a photo of the {c}.',
    ],
    "cars":[
        'a photo of a {c}.',
        'a photo of the {c}.',
        'a photo of my {c}.',
        'i love my {c}!',
        'a photo of my dirty {c}.',
        'a photo of my clean {c}.',
        'a photo of my new {c}.',
        'a photo of my old {c}.',
    ],
    "fgvc_aircraft":[
        'a photo of a {c}, a type of aircraft.',
        'a photo of the {c}, a type of aircraft.',
    ],
    "dtd":[
        'a photo of a {c} texture.',
        'a photo of a {c} pattern.',
        'a photo of a {c} thing.',
        'a photo of a {c} object.',
        'a photo of the {c} texture.',
        'a photo of the {c} pattern.',
        'a photo of the {c} thing.',
        'a photo of the {c} object.',    
    ],
    "pets":[
        'a photo of a {c}, a type of pet.',
    ],
    "caltech101":[
        'a photo of a {c}.',
        'a painting of a {c}.',
        'a plastic {c}.',
        'a sculpture of a {c}.',
        'a sketch of a {c}.',
        'a tattoo of a {c}.',
        'a toy {c}.',
        'a rendition of a {c}.',
        'a embroidered {c}.',
        'a cartoon {c}.',
        'a {c} in a video game.',
        'a plushie {c}.',
        'a origami {c}.',
        'art of a {c}.',
        'graffiti of a {c}.',
        'a drawing of a {c}.',
        'a doodle of a {c}.',
        'a photo of the {c}.',
        'a painting of the {c}.',
        'the plastic {c}.',
        'a sculpture of the {c}.',
        'a sketch of the {c}.',
        'a tattoo of the {c}.',
        'the toy {c}.',
        'a rendition of the {c}.',
        'the embroidered {c}.',
        'the cartoon {c}.',
        'the {c} in a video game.',
        'the plushie {c}.',
        'the origami {c}.',
        'art of the {c}.',
        'graffiti of the {c}.',
        'a drawing of the {c}.',
        'a doodle of the {c}.',
    ],
    "flowers":[
        'a photo of a {c}, a type of flower.',
    ],
    "mnist": [
        'a photo of the number: "{c}".',
    ],
    "stl10": [
        'a photo of a {c}.',
        'a photo of the {c}.',
    ],
    "eurosat":[
        'a centered satellite photo of {c}.',
        'a centered satellite photo of a {c}.',
        'a centered satellite photo of the {c}.',
    ],
    "gtsrb":[
        'a zoomed in photo of a "{c}" traffic sign.',
        'a centered photo of a "{c}" traffic sign.',
        'a close up photo of a "{c}" traffic sign.',
    ],
    "kitti":[
        '{c}',
    ],
    "country211":[
        'a photo i took in {c}.',
        'a photo i took while visiting {c}.',
        'a photo from my home country of {c}.',
        'a photo from my visit to {c}.',
        'a photo showing the country of {c}.',
    ],
    "pcam":[
        '{c}',
    ],
    "renderedsst2":[
        'a {c} review of a movie.',
    ],
    "voc2007":[
        'a photo of a {c}.',
    ]
}


# OxfordIIITPet  from torchvision does not have a classes attribute, so we define the classes here
flowers_classes = [
    'pink primrose',
    'hard-leaved pocket orchid',
    'canterbury bells',
    'sweet pea',
    'english marigold',
    'tiger lily',
    'moon orchid',
    'bird of paradise',
    'monkshood',
    'globe thistle',
    'snapdragon',
    "colt's foot",
    'king protea',
    'spear thistle',
    'yellow iris',
    'globe flower',
    'purple coneflower',
    'peruvian lily',
    'balloon flower',
    'giant white arum lily',
    'fire lily',
    'pincushion flower',
    'fritillary',
    'red ginger',
    'grape hyacinth',
    'corn poppy',
    'prince of wales feathers',
    'stemless gentian',
    'artichoke',
    'sweet william',
    'carnation',
    'garden phlox',
    'love in the mist',
    'mexican aster',
    'alpine sea holly',
    'ruby-lipped cattleya',
    'cape flower',
    'great masterwort',
    'siam tulip',
    'lenten rose',
    'barbeton daisy',
    'daffodil',
    'sword lily',
    'poinsettia',
    'bolero deep blue',
    'wallflower',
    'marigold',
    'buttercup',
    'oxeye daisy',
    'common dandelion',
    'petunia',
    'wild pansy',
    'primula',
    'sunflower',
    'pelargonium',
    'bishop of llandaff',
    'gaura',
    'geranium',
    'orange dahlia',
    'pink and yellow dahlia',
    'cautleya spicata',
    'japanese anemone',
    'black-eyed susan',
    'silverbush',
    'californian poppy',
    'osteospermum',
    'spring crocus',
    'bearded iris',
    'windflower',
    'tree poppy',
    'gazania',
    'azalea',
    'water lily',
    'rose',
    'thorn apple',
    'morning glory',
    'passion flower',
    'lotus',
    'toad lily',
    'anthurium',
    'frangipani',
    'clematis',
    'hibiscus',
    'columbine',
    'desert-rose',
    'tree mallow',
    'magnolia',
    'cyclamen',
    'watercress',
    'canna lily',
    'hippeastrum',
    'bee balm',
    'air plant',
    'foxglove',
    'bougainvillea',
    'camellia',
    'mallow',
    'mexican petunia',
    'bromelia',
    'blanket flower',
    'trumpet creeper',
    'blackberry lily',
]

gtsrb_classes = [
    'red and white circle 20 kph speed limit',
    'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit',
    'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit',
    'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit',
    'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit',
    'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing',
    'red and white triangle road intersection warning',
    'white and yellow diamond priority road',
    'red and white upside down triangle yield right-of-way',
    'stop',
    'empty red and white circle',
    'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry',
    'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning',
    'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning',
    'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning',
    'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit',
    'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory',
    'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory',
    'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]

country211_classes = [
    'Andorra',
    'United Arab Emirates',
    'Afghanistan',
    'Antigua and Barbuda',
    'Anguilla',
    'Albania',
    'Armenia',
    'Angola',
    'Antarctica',
    'Argentina',
    'Austria',
    'Australia',
    'Aruba',
    'Aland Islands',
    'Azerbaijan',
    'Bosnia and Herzegovina',
    'Barbados',
    'Bangladesh',
    'Belgium',
    'Burkina Faso',
    'Bulgaria',
    'Bahrain',
    'Benin',
    'Bermuda',
    'Brunei Darussalam',
    'Bolivia',
    'Bonaire, Saint Eustatius and Saba',
    'Brazil',
    'Bahamas',
    'Bhutan',
    'Botswana',
    'Belarus',
    'Belize',
    'Canada',
    'DR Congo',
    'Central African Republic',
    'Switzerland',
    "Cote d'Ivoire",
    'Cook Islands',
    'Chile',
    'Cameroon',
    'China',
    'Colombia',
    'Costa Rica',
    'Cuba',
    'Cabo Verde',
    'Curacao',
    'Cyprus',
    'Czech Republic',
    'Germany',
    'Denmark',
    'Dominica',
    'Dominican Republic',
    'Algeria',
    'Ecuador',
    'Estonia',
    'Egypt',
    'Spain',
    'Ethiopia',
    'Finland',
    'Fiji',
    'Falkland Islands',
    'Faeroe Islands',
    'France',
    'Gabon',
    'United Kingdom',
    'Grenada',
    'Georgia',
    'French Guiana',
    'Guernsey',
    'Ghana',
    'Gibraltar',
    'Greenland',
    'Gambia',
    'Guadeloupe',
    'Greece',
    'South Georgia and South Sandwich Is.',
    'Guatemala',
    'Guam',
    'Guyana',
    'Hong Kong',
    'Honduras',
    'Croatia',
    'Haiti',
    'Hungary',
    'Indonesia',
    'Ireland',
    'Israel',
    'Isle of Man',
    'India',
    'Iraq',
    'Iran',
    'Iceland',
    'Italy',
    'Jersey',
    'Jamaica',
    'Jordan',
    'Japan',
    'Kenya',
    'Kyrgyz Republic',
    'Cambodia',
    'St. Kitts and Nevis',
    'North Korea',
    'South Korea',
    'Kuwait',
    'Cayman Islands',
    'Kazakhstan',
    'Laos',
    'Lebanon',
    'St. Lucia',
    'Liechtenstein',
    'Sri Lanka',
    'Liberia',
    'Lithuania',
    'Luxembourg',
    'Latvia',
    'Libya',
    'Morocco',
    'Monaco',
    'Moldova',
    'Montenegro',
    'Saint-Martin',
    'Madagascar',
    'Macedonia',
    'Mali',
    'Myanmar',
    'Mongolia',
    'Macau',
    'Martinique',
    'Mauritania',
    'Malta',
    'Mauritius',
    'Maldives',
    'Malawi',
    'Mexico',
    'Malaysia',
    'Mozambique',
    'Namibia',
    'New Caledonia',
    'Nigeria',
    'Nicaragua',
    'Netherlands',
    'Norway',
    'Nepal',
    'New Zealand',
    'Oman',
    'Panama',
    'Peru',
    'French Polynesia',
    'Papua New Guinea',
    'Philippines',
    'Pakistan',
    'Poland',
    'Puerto Rico',
    'Palestine',
    'Portugal',
    'Palau',
    'Paraguay',
    'Qatar',
    'Reunion',
    'Romania',
    'Serbia',
    'Russia',
    'Rwanda',
    'Saudi Arabia',
    'Solomon Islands',
    'Seychelles',
    'Sudan',
    'Sweden',
    'Singapore',
    'St. Helena',
    'Slovenia',
    'Svalbard and Jan Mayen Islands',
    'Slovakia',
    'Sierra Leone',
    'San Marino',
    'Senegal',
    'Somalia',
    'South Sudan',
    'El Salvador',
    'Sint Maarten',
    'Syria',
    'Eswatini',
    'Togo',
    'Thailand',
    'Tajikistan',
    'Timor-Leste',
    'Turkmenistan',
    'Tunisia',
    'Tonga',
    'Turkey',
    'Trinidad and Tobago',
    'Taiwan',
    'Tanzania',
    'Ukraine',
    'Uganda',
    'United States',
    'Uruguay',
    'Uzbekistan',
    'Vatican',
    'Venezuela',
    'British Virgin Islands',
    'United States Virgin Islands',
    'Vietnam',
    'Vanuatu',
    'Samoa',
    'Kosovo',
    'Yemen',
    'South Africa',
    'Zambia',
    'Zimbabwe',
]

eurosat_classes = [
    'forest',
    'permanent crop land',
    'residential buildings or homes or apartments',
    'river',
    'pasture land',
    'lake or sea',
    'brushland or shrubland',
    'annual crop land',
    'industrial buildings or commercial buildings',
    'highway or road',
]