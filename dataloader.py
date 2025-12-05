import torch
from torchvision import datasets, transforms
import os
import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence
from PIL import Image
import warnings

warnings.filterwarnings(
    "ignore", 
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images", 
    category=UserWarning,
    module='PIL.Image'
)



data_transform = transforms.Compose([
    transforms.Resize(256),              # Resize smaller dimension to 256
    transforms.CenterCrop(224),          # Crop to 224x224 (standard input size)
    transforms.Pad(50, padding_mode="reflect"),
    transforms.ToTensor(),               # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])


class_cleanup_dict = {
"":                                                                                                                                   "",
"001_BlackKite":                                                                                                                      "blackkite",
"002_BRAHMINY_KITE":                                                                                                                  "brahminy kite",
"003_IndianPeaflow":                                                                                                                  "indianpeaflow",
"004_INDIAN_ROBIN":                                                                                                                   "indian robin",
"005_IndianRoller":                                                                                                                   "indianroller",
"006_LITTLE_EGRET":                                                                                                                   "little egret",
"007_ORIENTAL_MAGPIE_ROBIN":                                                                                                          "oriental magpie robin",
"008_SPOTTED_DOVE":                                                                                                                   "spotted dove",
"009_YellowFootedGreenPegion":                                                                                                        "yellowfootedgreenpegion",
"1,065_White_Breasted_Kingfisher_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                "white breasted kingfisher",
"1,124_Stone_Curlew_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                             "stone curlew",
"13,871_Coot_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                                    "coot",
"141,863_Grey_Treepie_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                           "grey treepie",
"166,298_Humes_Leaf_Warbler_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStockBoards":                                               "humes leaf warbler",
"2,291_Common_Kestrel_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                           "common kestrel",
"3,127_Black_Winged_Stilt_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                       "black winged stilt",
"4,303_Mynah_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                                    "mynah",
"4,555_Roller_Bird_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                              "roller bird",
"423_Long_Tailed_Shrike_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                         "long tailed shrike",
"469_Sand_Grouse_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                                "sand grouse",
"6_6_Great_Indian_Bustard_-_Ardeotis_nigriceps_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":"great indian bustard",
"771_Red_Wattled_Lapwing_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStockBoards":                                                  "red wattled lapwing",
"800+_Best_Eagle_Images_·_100_Free_Download_·_Pexels_Stock_Photos":                                                                   "eagle",
"804_Western_Marsh_Harrier_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                      "western marsh harrier,",
"99,740_Fairy_Bluebird_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                          "fairy bluebird,",
"Alexandrine_Parakeet":                                                                                                               "alexandrine parakeet",
"Alexandrine_Parakeet_-_Psittacula_eupatria_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":   "alexandrine parakeet",
"American_White_Pelican":                                                                                                             "american white pelican",
"American_White_Pelican_-_Pelecanus_erythrorhynchos_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Libr":   "american white pelican",
"Ashy_Prinia":                                                                                                                        "ashy prinia",
"Ashy_Wood-Pigeon_-_Columba_pulchricollis_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":     "ashy wood-pigeon",
"Asian Woolly-necked Stork":                                                                                                          "asian woolly-necked stork",
"Asian_Koel":                                                                                                                         "asian koel",
"BARN OWL":                                                                                                                           "barn owl",
"BULBUL":                                                                                                                             "bulbul",
"Black-and-yellow Grosbeak":                                                                                                          "black-and-yellow grosbeak",
"Black-headed_Ibis_-_Threskiornis_melanocephalus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library":   "black-headed ibis",
"Black_Drongo_-_Dicrurus_macrocercus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":          "black drongo",
"Black_Hooded_Oriole_Stock_Photos,_Pictures_Royalty-Free_Images_-_iStock":                                                            "black hooded oriole,",
"Blue-bearded_Bee-eater_-_Nyctyornis_athertoni_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_l":   "blue-bearded bee-eater",
"Brown-headed_Barbet_-_Psilopogon_zeylanicus_-_Media_Search_-_Macaulay_Library_and_eBird":                                            "brown-headed barbet",
"Brown_Bush_Warbler_-_Locustella_luteoventris_-_Media_Search_-_Macaulay_Library_and_eBird":                                           "brown bush warbler",
"Brown_Shrike_-_Lanius_cristatus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":              "brown shrike",
"Cape_Rock-Thrush_-_Monticola_rupestris_-_Media_Search_-_Macaulay_Library_and_eBird":                                                 "cape rock-thrush",
"Caspian_Gull":                                                                                                                       "caspian gull",
"Chilean_Flamingo_-_Phoenicopterus_chilensis_-_Media_Search_-_Macaulay_Library_and_eBird":                                            "chilean flamingo",
"Common_Iora_-_Aegithina_tiphia_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":               "common iora",
"Common_Shelduck_-_Tadorna_tadorna_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":            "common shelduck",
"Crab-Plover_-_Dromas_ardeola_-_Media_Search_-_Macaulay_Library_and_eBird":                                                           "crab-plover",
"EGYPTIAN GOOSE":                                                                                                                     "egyptian goose",
"Eurasian_Hoopoe_-_Upupa_epops_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":                "eurasian hoopoe",
"Great Cormorant":                                                                                                                    "great cormorant",
"Great Horned Owl":                                                                                                                   "great horned owl",
"Greater_Coucal_-_Centropus_sinensis_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":          "greater coucal",
"Grey headed swamphen":                                                                                                               "grey headed swamphen",
"Hill_Swallow_-_Hirundo_domicola_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":              "hill swallow",
"House_Sparrow_(Indian)_-_Passer_domesticus_[indicus_Group]_-_Media_Search_-_Macaulay_Library_and_eBird":                             "house sparrow (indian)",
"House_Sparrow_-_Passer_domesticus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":            "house sparrow",
"Indian Courser":                                                                                                                     "indian courser",
"Indian Paradise-Flycatcher":                                                                                                         "indian paradise-flycatcher",
"Indian Spot-billed Duck":                                                                                                            "indian spot-billed duck",
"Indian_Bushlark_-_Mirafra_erythroptera_-_Media_Search_-_Macaulay_Library_and_eBird":                                                 "indian bushlark",
"Indian_Cormorant_-_Phalacrocorax_fuscicollis_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_lo":   "indian cormorant",
"Indian_Cuckoo_-_Cuculus_micropterus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":          "indian cuckoo",
"Indian_Pond-Heron_-_Ardeola_grayii_-_Media_Search_-_Macaulay_Library_and_eBird":                                                     "indian pond-heron",
"Indian_Vulture_-_Gyps_indicus_-_Media_Search_-_Macaulay_Library_and_eBird":                                                          "indian vulture",
"Jerdon_s_Leafbird_-_Chloropsis_jerdoni_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":       "jerdon leafbird",
"Large-billed_Crow_(Indian_Jungle)_-_Corvus_macrorhynchos_culminatus_-_Media_Search_-_Macaulay_Library_and_eBird":                    "large-billed crow",
"Large_Gray_Babbler_-_Argya_malcolmi_-_Media_Search_-_Macaulay_Library_and_eBird":                                                    "large gray babbler",
"Laughing Dove":                                                                                                                      "laughing dove",
"Lesser Fish-Eagle":                                                                                                                  "lesser fish-eagle",
"Lesser_Flamingo_-_Phoeniconaias_minor_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":        "lesser flamingo",
"Little Cormorant":                                                                                                                   "little cormorant",
"Little Ringed Plover":                                                                                                               "little ringed plover",
"Malabar_Trogon_-_Harpactes_fasciatus_-_Media_Search_-_Macaulay_Library_and_eBird":                                                   "malabar trogon",
"Malabar_Whistling-Thrush_-_Myophonus_horsfieldii_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Librar":   "malabar whistling-thrush",
"Nicobar_Pigeon_-_Caloenas_nicobarica_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":         "nicobar pigeon",
"Nilgiri_Flycatcher_-_Eumyias_albicaudatus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":    "nilgiri flycatcher",
"Nilgiri_Pipit_-_Anthus_nilghiriensis_-_Media_Search_-_Macaulay_Library_and_eBird":                                                   "nilgiri pipit",
"Northern Shoveler - Spatula clypeata":                                                                                               "northern shoveler",
"Northern_Pintail_-_Anas_acuta_-_Media_Search_-_Macaulay_Library_and_eBird":                                                          "northern pintail",
"Oriental_Pratincole_-_Glareola_maldivarum_-_Media_Search_-_Macaulay_Library_and_eBird":                                              "oriental pratincole",
"Paddyfield_Warbler_-_Acrocephalus_agricola_-_Media_Search_-_Macaulay_Library_and_eBird":                                             "paddyfield warbler",
"Painted_Bush-Quail_-_Perdicula_erythrorhyncha_-_Media_Search_-_Macaulay_Library_and_eBird":                                          "painted bush-quail",
"Painted_Stork_-_Mycteria_leucocephala_-_Media_Search_-_Macaulay_Library_and_eBird":                                                  "painted stork",
"Pallas's Gull":                                                                                                                      "pallas's gull",
"Peregrine_Falcon_-_Falco_peregrinus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":          "peregrine falcon",
"Pheasant-tailed Jacana":                                                                                                             "pheasant-tailed jacana",
"Plain prinia":                                                                                                                       "plain prinia",
"Purple-Sunbird":                                                                                                                     "purple-sunbird",
"RED BEARDED BEE EATER":                                                                                                              "red bearded bee eater",
"Red-vented_Bulbul_-_Pycnonotus_cafer_-_Media_Search_-_Macaulay_Library_and_eBird":                                                   "red-vented bulbul",
"Red_Junglefowl_-_Gallus_gallus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":               "red junglefowl",
"Red_Spurfowl_-_Galloperdix_spadicea_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":          "red spurfowl",
"Richard_s_Paddyfield_Pipit_-_Anthus_richardi_rufulus_-_Media_Search_-_Macaulay_Library_and_eBird":                                   "richard paddyfield pipit",
"Royal_Spoonbill_-_Platalea_regia_-_Media_Search_-_Macaulay_Library_and_eBird":                                                       "royal spoonbill",
"Rufous_Treepie_-_Dendrocitta_vagabunda_-_Media_Search_-_Macaulay_Library_and_eBird":                                                 "rufous treepie",
"Sarus_Crane_-_Antigone_antigone_-_Media_Search_-_Macaulay_Library_and_eBird":                                                        "sarus crane",
"Scaly breasted munia":                                                                                                               "scaly breasted munia",
"Shikra_-_Accipiter_badius_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":                    "shikra",
"Short-toed_Snake-Eagle_-_Circaetus_gallicus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_log":   "short-toed snake-eagle",
"Siberian_Crane_-_Leucogeranus_leucogeranus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":   "siberian crane",
"Silver-breasted Broadbill":                                                                                                          "silver-breasted broadbill",
"Spot-billed_Pelican_-_Pelecanus_philippensis_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_lo":   "spot-billed pelican",
"Spotted_Elachura_-_Elachura_formosa_-_Media_Search_-_Macaulay_Library_and_eBird":                                                    "spotted elachura",
"Stock_Dove_-_Columba_oenas_-_Media_Search_-_Macaulay_Library_and_eBird":                                                             "stock dove",
"Tawny-bellied_Babbler_-_Dumetia_hyperythra_-_Media_Search_-_Macaulay_Library_and_eBird":                                             "tawny-bellied babbler",
"Tundra_Swan_-_Cygnus_columbianus_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":             "tundra swan",
"WOODPEAKER":                                                                                                                         "woodpeaker",
"WREN":                                                                                                                               "wren",
"White-eyed_Buzzard_-_Butastur_teesa_-_Media_Search_-_Macaulay_Library_and_eBird":                                                    "white-eyed buzzard",
"Yellow breasted greenfinch":                                                                                                         "yellow breasted greenfinch",
"Yellow-billed_Barbet_-_Trachyphonus_purpuratus_-_Media_Search_-_Macaulay_Library_and_eBird":                                         "yellow-billed barbet",
"Yellow_Bittern_-_Ixobrychus_sinensis_-_Media_Search_-_Macaulay_Library_and_eBirdMacaulay_Library_logoMacaulay_Library_logo":         "yellow bittern",
"pegion-gullimeot":                                                                                                                   "pegion-gullimeot",
}

def read_folder(path : str) -> list[str]:
    image_paths = []
    for name in os.listdir(path):
        item_path = os.path.join(path, name)
        if os.path.isdir(item_path):
            image_paths += read_folder(item_path)
        else:
            image_paths.append(item_path)
    return image_paths
class MergedImageFolder(datasets.DatasetFolder):
    def __init__(self, root, transform=None, use_fraction : float | tuple[float, float] = 1, dataset_nr = 1):
        self.data_pairs : list[tuple[str, str]] = [] # tuples of (path, class)
        self.transform = transform
        classes = set()
        
        for sub_dir in os.listdir(root):
            sub_path = os.path.join(root, sub_dir, "raw") if dataset_nr == 1 else os.path.join(root, sub_dir)
            if os.path.isdir(sub_path):
                if dataset_nr == 1:
                    image_dirs = [obj for obj in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, obj))]
                    for dir in image_dirs:
                        base_dir = os.path.join(sub_path, dir)
                        image_class = class_cleanup_dict.get(dir, dir)
                        classes.add(image_class)
                        image_paths = read_folder(base_dir)
                        [self.data_pairs.append((path, image_class)) for path in image_paths if path.endswith((".jpg", ".png", "jpeg"))]
                else:
                    image_class = class_cleanup_dict.get(sub_dir, sub_dir)
                    classes.add(image_class)
                    image_paths = read_folder(sub_path)
                    [self.data_pairs.append((path, image_class)) for path in image_paths if path.endswith((".jpg", ".png", "jpeg"))]


        if isinstance(use_fraction, (float, int)):
            use_fraction = (0, use_fraction)
        values = np.arange(len(self.data_pairs))/len(self.data_pairs)
        np.random.shuffle(values)
        rs = RandomState(MT19937(SeedSequence(123456789)))
        self.data_pairs = np.array(self.data_pairs)
        self.data_pairs = self.data_pairs[(values >= use_fraction[0]) & (values < use_fraction[1])]
        self.classes = list(classes)
        self.classes.sort()
        self.classes.insert(0, "")
        self.class_to_idx = {image_class : i for i, image_class in enumerate(self.classes)}
        #self.class_to_idx[""] = 0 # no class


    

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # returns the image and the class index
        path, image_class = self.data_pairs[idx]
        image = Image.open(path).convert("RGB")#.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[image_class]

if __name__ == "__main__":
    bird_dataset1 = MergedImageFolder(root="data1", transform=data_transform, use_fraction=(0.1, 0.8), dataset_nr=1)
    bird_dataset2 = MergedImageFolder(root="data2/CUB_200_2011/images/", transform=data_transform, use_fraction=(0.1, 0.8), dataset_nr=2)

    if False:
        pres = []
        posts = []
        for i, clas in enumerate(bird_dataset1.classes):
            clas : str
            pre = f"\"{clas}\":"
            post = clas.lower().replace("_macaulay_library_and_ebirdmacaulay", "").replace("_media_search_", "") \
            .replace("_istock", "").replace("_istockboards", "").replace("_stock_photos", "").replace("_pictures_royalty-free_images_", "") \
            .replace("_", " ").replace("macaulay library and ebird", "").replace("library logomacaulay library", "").strip("- ")
            post = "\"" + post + "\","
            pres.append(pre)
            posts.append(post)
        max_length = max(map(len, pres))
        for pre, post in zip(pres, posts):
            print(pre + (" " * (max_length - len(pre))) + post)
    
        
    # try load all items
    for i, dataset in enumerate([bird_dataset1, bird_dataset2]):
        class_counts = [0 for i in range(len(dataset.classes))]
        for image, class_id in dataset:
            class_counts[class_id] += 1
        print(f"Dataset {i}:")
        class_average = float(np.mean(class_counts))
        class_spread = float(np.std(class_counts))
        print(f"  number of entries: {len(dataset)}")
        print(f"  number of classes: {len(dataset.classes)}")
        print(f"  class count average, spread: {class_average:.3f} {class_spread:.3f}")


