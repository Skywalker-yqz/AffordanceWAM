WAN_WRAPPED_MODULE = "affordance_wam.wan_vendor.shared.core.vram.layers.AutoWrappedModule"
WAN_WRAPPED_LINEAR = "affordance_wam.wan_vendor.shared.core.vram.layers.AutoWrappedLinear"
WAN_WRAPPED_NON_RECURSE = "affordance_wam.wan_vendor.shared.core.vram.layers.AutoWrappedNonRecurseModule"

VRAM_MANAGEMENT_MODULE_MAPS = {
    "affordance_wam.wan_vendor.models.dit.WanModel": {
        "affordance_wam.wan_vendor.models.dit.MLP": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.dit.DiTBlock": WAN_WRAPPED_NON_RECURSE,
        "affordance_wam.wan_vendor.models.dit.Head": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv3d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.dit.RMSNorm": WAN_WRAPPED_MODULE,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
    },
    "affordance_wam.wan_vendor.models.image_encoder.WanImageEncoder": {
        "affordance_wam.wan_vendor.models.image_encoder.VisionTransformer": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
    },
    "affordance_wam.wan_vendor.models.text_encoder.WanTextEncoder": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Embedding": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.text_encoder.T5RelativeEmbedding": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.text_encoder.T5LayerNorm": WAN_WRAPPED_MODULE,
    },
    "affordance_wam.wan_vendor.models.vace.VaceWanModel": {
        "affordance_wam.wan_vendor.models.dit.DiTBlock": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv3d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.dit.RMSNorm": WAN_WRAPPED_MODULE,
    },
    "affordance_wam.wan_vendor.models.vae.WanVideoVAE": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.vae.RMS_norm": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.vae.CausalConv3d": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.vae.Upsample": WAN_WRAPPED_MODULE,
        "torch.nn.SiLU": WAN_WRAPPED_MODULE,
        "torch.nn.Dropout": WAN_WRAPPED_MODULE,
    },
    "affordance_wam.wan_vendor.models.vae.WanVideoVAE38": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.vae.RMS_norm": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.vae.CausalConv3d": WAN_WRAPPED_MODULE,
        "affordance_wam.wan_vendor.models.vae.Upsample": WAN_WRAPPED_MODULE,
        "torch.nn.SiLU": WAN_WRAPPED_MODULE,
        "torch.nn.Dropout": WAN_WRAPPED_MODULE,
    },
}
