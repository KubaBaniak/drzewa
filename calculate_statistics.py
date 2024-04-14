import numpy as np
from PIL import Image


def calculate_tree_planting_range(area_km2):
    area_ha = area_km2 * 100

    tree_species = {
        'Sosna': (8000, 10000),
        'Świerk': (3000, 5000),
        'Jodła': (4000, 8000),
        'Modrzew': (1500, 3000),
        'Jedlica': (3000, 4000),
        'Dąb': (6000, 8000),
        'Buk': (6000, 8000),
        'Inne liściaste': (4000, 6000)
    }

    planting_ranges = {}
    for species, (min_per_ha, max_per_ha) in tree_species.items():
        min_trees = round(area_ha * min_per_ha)
        max_trees = round(area_ha * max_per_ha)
        planting_ranges[species] = (min_trees, max_trees)

    return planting_ranges


def calculate_forest_statistics(image):
    forest_mask = image == 255

    total_pixels = image.size
    forest_pixels = np.sum(forest_mask)
    forest_percentage = (forest_pixels / total_pixels) * 100

    pixel_area_km2 = (0.86 / 1024) ** 2
    total_area = pixel_area_km2 * 1024 ** 2
    forest_area_km2 = forest_pixels * pixel_area_km2

    deforested_area_km2 = total_area - forest_area_km2

    return forest_percentage, forest_area_km2, deforested_area_km2


def display_stats(statistics):
    print(f"Procent zalesionych terenów: {statistics[0]:.2f}%")
    print(f"Powierzchnia lasu: {statistics[1]:.2f} km²")
    print(f"Powierzchnia terenów wylesionych: {statistics[2]:.2f} km²\n")


print('Kwiecień 2019r.')
pil_mask1 = Image.open('puszcza_notecka/calculated_masks/mask12.png').convert('L')
mask1 = np.array(pil_mask1)
stats1 = calculate_forest_statistics(mask1)
display_stats(stats1)

pil_mask2 = Image.open('puszcza_notecka/calculated_masks/mask11.png').convert('L')
print('Maj 2021r.')
mask2 = np.array(pil_mask2)
stats2 = calculate_forest_statistics(mask2)
display_stats(stats2)

area_km2 = stats1[1] - stats2[1]
planting_ranges = calculate_tree_planting_range(area_km2)
print(f'Ilość potrzebnych sadzonek do odnowienia sztucznego lasu o powierzchni {area_km2:.2f} km²')
for species, ranges in planting_ranges.items():
    print(f"{species}: Od {ranges[0]} do {ranges[1]} sztuk")
