"""CatTemplate modules."""

from opencat3d.template.cat_template import (
    CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES,
    CatTemplate,
    CatTemplateRegion,
    CatTemplateRegionUvIsland,
    CatTemplateRegionV2,
    create_placeholder_cat_template,
    load_cat_template_npz,
    validate_cat_template_region_manifest_v2,
)

__all__ = [
    "CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES",
    "CatTemplate",
    "CatTemplateRegion",
    "CatTemplateRegionUvIsland",
    "CatTemplateRegionV2",
    "create_placeholder_cat_template",
    "load_cat_template_npz",
    "validate_cat_template_region_manifest_v2",
]
