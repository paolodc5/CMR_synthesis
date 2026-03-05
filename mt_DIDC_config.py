GROUPING_RULES = {
    "Background": "Background",
    "__unused__": "Others",
    "Other_tissue": "Others",
    "Cartilage_costal": "Bones_and_cartilage",
    "Humerus_left": "Bones_and_cartilage",
    "Scapula_left": "Bones_and_cartilage",
    "Scapula_right": "Bones_and_cartilage",
    "Clavicle_left": "Bones_and_cartilage",
    "Clavicle_right": "Bones_and_cartilage",
    "Pelvis_right": "Bones_and_cartilage",
    "Sternum": "Bones_and_cartilage",
    "Cartilage_costal": "Bones_and_cartilage",
    "Intervertebral_disc": "Bones_and_cartilage",
    "Vertebrae_body": "Bones_and_cartilage",
    "Vertebrae_posterior_processes": "Bones_and_cartilage",
    "Spinal_canal": "Bones_and_cartilage",
    "Kidney_right": "Kidneys",
    "Kidney_left":  "Kidneys",
    "Adrenal_gland_right": "Kidneys",
    "Adrenal_gland_left": "Kidneys",
    "Lung": "Lungs",
    "Vein_pulmonary": "Blood_vessels",
    "Artery_brachiocephalic": "Blood_vessels",
    "Artery_subclavian_right": "Blood_vessels",
    "Artery_subclavian_left": "Blood_vessels",
    "Artery_common_carotid_right": "Blood_vessels",
    "Artery_common_carotid_left": "Blood_vessels",
    "Vein_brachiocephalic_left": "Blood_vessels",
    "Vein_brachiocephalic_right": "Blood_vessels",
    "Vena_cava_superior": "Blood_vessels",
    "Vena_cava_inferior": "Blood_vessels",
    "Vein_portal_and_splenic": "Blood_vessels",
    "Stomach_wall": "Stomach",
    "Stomach_lumen": "Stomach",
    "Esophagus_lumen": "Esophagus_and_trachea",
    "Esophagus_wall": "Esophagus_and_trachea",
    "Trachea_wall": "Esophagus_and_trachea",
    "Trachea_lumen": "Esophagus_and_trachea",
    "Thyroid_gland": "Esophagus_and_trachea",
    "Intestine_lumen": "Intestine",
    "Small_intestine_duodenum": "Intestine",
    "Intestine_wall": "Intestine",
    "Gallbladder_wall": "Gallbladder",
    "Gallbladder_bile": "Gallbladder",
    "Fat": "Fat",
    "SAT": "Fat",
    "Muscle": "Muscle",
    "Skin": "Skin",
    "Spleen": "Spleen",
    "Pancreas": "Pancreas",
    "Heart": "Heart_generic",
    "Atrial_appendage_left": "Heart_generic",
    "Liver": "Liver",
    "Aorta": "Aorta",
    "Spleen": "Spleen",
    "Spinal_cord": "Spinal_cord",
}

# ITIS database values for reference
PROPERTY_KEY = {
    "blood": [80.43, 1414.8, 308.5],  # (ITIS)
    "bone": [15.30, 288.0, 165.0], # (ITIS yellow marrow)
    "muscle": [76.21, 981.5, 36.0], # (ITIS)
    "fat": [100., 288.0, 165.0],  # (Gold et al. 2012, subcutaneous fat)
    "heart_muscle": [79.47, 1026.3, 42.0],  # (ITIS)
    "liver": [76.33, 661.5, 56.8],  # (ITIS)
    "lungs": [27.20, 1196., 6.3],  # (ITIS, T1 is from Dietrich et al 2016)
    "pancreas": [73.18, 584.0, 46.0], # (ITIS)
    "spleen": [79.34, 1057.0, 79.0], # (ITIS)
    "kidney": [80.50, 828.0,  71.0], # (ITIS)
    "spinal_cord": [71.84, 745.0, 74.0], # (ITIS)
    "skin": [65.19, 900.0, 20.0], # (ITIS, T1 and T2 are estimated from derma values in Richard et al, 1991)
    "others": [71.0, 250.0, 20.0], # from Buoso
    "background_air": [0.01, 750.0, 60.0], # from Buoso
    "trachea": [60.00, 1045.5, 37.3]
}
# "cartilage": [71.77, 1045.5, 37.3], # (ITIS)

LABEL2LABEL = {
    "Aorta": "blood",
    "Background": "background_air",
    "Blood_vessels": "blood",
    "Bones_and_cartilage": "bone",
    "Esophagus_and_trachea": "trachea",
    "Fat": "fat",
    "Gallbladder": "others",
    "Heart_generic": "heart_muscle",
    "Intestine": "others",
    "Kidneys": "kidney",
    "LV_Myocardium": "heart_muscle",
    "LV_blood_pool": "blood",
    "Liver": "liver",
    "Lungs": "lungs",
    "Muscle": "muscle",
    "Others": "others",
    "Pancreas": "pancreas",
    "RV_blood_pool_myocardium": "heart_muscle",
    "Skin": "skin",
    "Spinal_cord": "spinal_cord",
    "Spleen": "spleen",
    "Stomach": "muscle",
}

new_labels = set()
for value in GROUPING_RULES.values():
    new_labels.add(value)

foreground_labels = ["LV_Myocardium", "LV_blood_pool", "RV_blood_pool_myocardium"]
new_labels = new_labels.union(set(foreground_labels))

NEW_LABELS = sorted(list(new_labels))
