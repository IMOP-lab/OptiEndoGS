import torch
from torch.nn import functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.white_balance import simple_color_balance


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, separate_sh=False,
           override_color=None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True,
                                     device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    features_dc = pc.get_features_dc
    features_rest = pc.get_features_rest


    use_avg_appearance = True

    R = viewpoint_camera.world_view_transform[:3, :3].flatten()  
    T = viewpoint_camera.world_view_transform[3, :3]

    cam_pose = torch.cat((R, T), dim=-1).cuda()
    pose_features = pc.position_encoding(cam_pose)

    pose_emd = pc.embeddings(pose_features)


    appearance_embed = pose_emd
    appearance_embeddings = pose_emd.repeat(means3D.shape[0], 1)

    dir_pp_normalized = F.normalize(pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1),
                                    dim=1)
    embeddings = pc._get_fourier_features(pc.get_xyz, num_features=4)
    embeddings.add_(torch.randn_like(embeddings) * 0.001)

    position = pc.position_encoding(dir_pp_normalized)
    embed = torch.cat((appearance_embeddings, position), dim=-1)



    app_features = pc.appearance_mlp(features_dc.squeeze(1), appearance_embeddings, embeddings).clamp_max(1.0)

    colors_toned = torch.cat((app_features[:, None, :], features_rest), dim=1)

    shdim = (3 + 1) ** 2
    colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
    colors_toned = eval_sh(3, colors_toned, dir_pp_normalized)
    colors_toned = torch.clamp_min(colors_toned + 0.5, 0.0)

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    shs = None
    colors_precomp = colors_toned

    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)


    depth_z = 1 / (depth_image + 1e-5) 


    H, W = rendered_image.shape[1], rendered_image.shape[2]  


    depth_z = depth_z.squeeze()  

    appearance_embed_2d = appearance_embed.unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)

    attenuation_map = pc.attenuation_net(depth_z, appearance_embed_2d)    
    illumination_map = pc.illumination_net(depth_z)                       
    scattering_map = pc.scattering_net(depth_z)
    image_final = rendered_image * attenuation_map * illumination_map + scattering_map

    image_final = torch.clamp(image_final, 0.0, 1.0)


    rendered_J = simple_color_balance(rendered_image, scale=0.01, alpha=0.1)
    rendered_J = rendered_J.clamp(0, 1)

    out = {
        "render": image_final,         
        "restore_scene": rendered_J,  
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
        "illumination": illumination_map, 
        "attenuation": attenuation_map,
        "scattering": scattering_map,
    }

    return out