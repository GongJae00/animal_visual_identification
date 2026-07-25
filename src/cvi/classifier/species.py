from __future__ import annotations

CANIDAE_TAXONOMY: dict[str, list[str]] = {
    "canis_lupus_familiaris": [
        "affenpinscher", "afghan_hound", "airedale_terrier", "akita",
        "alaskan_malamute", "american_bulldog", "american_cocker_spaniel",
        "american_foxhound", "american_pit_bull_terrier", "american_staffordshire_terrier",
        "australian_cattle_dog", "australian_shepherd", "australian_terrier",
        "basenji", "basset_hound", "beagle", "bearded_collie",
        "bedlington_terrier", "belgian_malinois", "belgian_shepherd",
        "belgian_tervuren", "bernese_mountain_dog", "bichon_frise",
        "black_and_tan_coonhound", "bloodhound", "border_collie",
        "border_terrier", "borzoi", "boston_terrier", "bouvier_des_flandres",
        "boxer", "boykin_spaniel", "briard", "brittany_spaniel",
        "brussels_griffon", "bull_terrier", "bullmastiff", "cairn_terrier",
        "canaan_dog", "cane_corso", "cardigan_welsh_corgi", "cavalier_king_charles_spaniel",
        "chesapeake_bay_retriever", "chihuahua", "chinese_crested", "chinese_shar_pei",
        "chow_chow", "clumber_spaniel", "cocker_spaniel", "collie",
        "curly_coated_retriever", "dachshund", "dalmatian", "dandie_dinmont_terrier",
        "doberman_pinscher", "dogue_de_bordeaux", "english_cocker_spaniel",
        "english_foxhound", "english_setter", "english_springer_spaniel",
        "english_toy_spaniel", "entlebucher_mountain_dog", "field_spaniel",
        "finnish_lapphund", "finnish_spitz", "flat_coated_retriever",
        "french_bulldog", "german_pinscher", "german_shepherd_dog",
        "german_shorthaired_pointer", "german_wirehaired_pointer", "giant_schnauzer",
        "glen_of_imaal_terrier", "golden_retriever", "gordon_setter",
        "great_dane", "great_pyrenees", "greater_swiss_mountain_dog",
        "greyhound", "havanese", "ibizan_hound", "icelandic_sheepdog",
        "irish_setter", "irish_terrier", "irish_water_spaniel", "irish_wolfhound",
        "italian_greyhound", "japanese_chin", "keeshond", "kerry_blue_terrier",
        "komondor", "kuvasz", "labrador_retriever", "lago_romagnolo",
        "lakeland_terrier", "leonberger", "lhasa_apso", "lowchen",
        "maltese", "manchester_terrier", "mastiff", "miniature_bull_terrier",
        "miniature_pinscher", "miniature_poodle", "miniature_schnauzer",
        "neapolitan_mastiff", "newfoundland", "norfolk_terrier", "norwegian_buhund",
        "norwegian_elkhound", "norwegian_lundehund", "norwich_terrier",
        "nova_scotia_duck_tolling_retriever", "old_english_sheepdog", "otterhound",
        "papillon", "parson_russell_terrier", "pekingese", "pembroke_welsh_corgi",
        "petit_basset_griffon_vendeen", "pharaoh_hound", "plott_hound",
        "pointer", "polish_lowland_sheepdog", "pomeranian", "poodle",
        "portuguese_water_dog", "pug", "puli", "rhodesian_ridgeback",
        "rottweiler", "russian_toy", "saint_bernard", "saluki",
        "samoyed", "schipperke", "scottish_deerhound", "scottish_terrier",
        "sealyham_terrier", "shetland_sheepdog", "shiba_inu", "shih_tzu",
        "siberian_husky", "silky_terrier", "skye_terrier", "smooth_fox_terrier",
        "soft_coated_wheaten_terrier", "spanish_water_dog", "spinone_italiano",
        "staffordshire_bull_terrier", "standard_poodle", "standard_schnauzer",
        "sussex_spaniel", "swedish_vallhund", "tibetan_mastiff", "tibetan_spaniel",
        "tibetan_terrier", "toy_fox_terrier", "toy_poodle", "toy_manchester_terrier",
        "treeing_walker_coonhound", "vizsla", "weimaraner", "welsh_springer_spaniel",
        "welsh_terrier", "west_highland_white_terrier", "whippet",
        "wire_fox_terrier", "wirehaired_pointing_griffon", "xoloitzcuintli",
        "yorkshire_terrier",
    ],
}


def family_from_breed(breed: str) -> str:
    for family, breeds in CANIDAE_TAXONOMY.items():
        if breed in breeds:
            return family
    return "canis_lupus_familiaris"
