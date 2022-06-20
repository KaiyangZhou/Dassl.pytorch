import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY

from .wilds_base import WILDSBase

CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field", "dam",
    "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo"
]


@DATASET_REGISTRY.register()
class FMoW(WILDSBase):
    """Satellite imagery classification.

    62 classes (building or land use categories).

    Reference:
        - Christie et al. "Functional Map of the World." CVPR 2018.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "fmow_v1.1"

    def __init__(self, cfg):
        super().__init__(cfg)

    def get_image_path(self, dataset, idx):
        idx = dataset.full_idxs[idx]
        image_name = f"rgb_img_{idx}.png"
        image_path = osp.join(self.dataset_dir, "images", image_name)
        return image_path

    def get_domain(self, dataset, idx):
        # number of regions: 5 or 6
        # number of years: 16
        region_id = int(dataset.metadata_array[idx][0])
        year_id = int(dataset.metadata_array[idx][1])
        return region_id*16 + year_id

    def load_classnames(self):
        return {i: cat for i, cat in enumerate(CATEGORIES)}
