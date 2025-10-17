#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from .renderer import *
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh
from pathlib import Path
from internal.models.ciga_mlp import CigaMLP
import torch.nn as nn

class CigaRenderer(Renderer):
    def __init__(self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False):
        super().__init__()

        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            render_types: list = None
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if render_types is None:
            render_types = ["rgb"]
        assert len(render_types) == 1, "Only single type is allowed currently"

        rendered_image_key = "render"
        if "depth" in render_types:
            rendered_image_key = "depth"
            w2c = viewpoint_camera.world_to_camera  # already transposed
            means3D_in_camera_space = torch.matmul(pc.get_xyz, w2c[:3, :3]) + w2c[3, :3]
            depth = means3D_in_camera_space[:, 2:]
            # bg_color = torch.ones_like(bg_color) * depth.max()
            bg_color = torch.zeros_like(bg_color)
            override_color = depth.repeat(1, 3)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # 추가 
        shs = self.shs_weight_MLP(
            rasterizer, 
            shs, 
            viewpoint_camera, 
            means3D, 
            L=pc.max_sh_degree
            )
        
        #test----------------------------------------------------------------
        #cams = viewpoint_camera
        #cam_pos = cams.camera_center
        """
        print(f"cam_pos:{cam_pos.shape}\n"
              f"{cam_pos}\n"
              f"w2c:{cams.world_to_camera.shape}\n"
              f"{cams.world_to_camera}\n")
                     ->  campos는 카메라 중심 좌표 (배치가 1이라 shape이 3인거임)
                            w2c의 왼쪽 상단 3*3은 회전 R, 3행 0열~2열 : t
                             => 월드좌표계에서 카메라 좌표계로 변환할 때 사용
        am_pos:torch.Size([3])
        tensor([-5.5859e-03, -1.8000e+00, -2.1062e-13], device='cuda:0') 
        w2c:torch.Size([4, 4])
        tensor([[ 2.2204e-16,  0.0000e+00, -1.0000e+00,  0.0000e+00],
                [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00],
                [ 1.0000e+00,  0.0000e+00,  2.2204e-16,  0.0000e+00],
                [ 2.1062e-13,  1.8000e+00, -5.5859e-03,  1.0000e+00]], device='cuda:0')
        """
    
        #print("means: ", means3D.shape, means3D)
        #test----------------------------------------------------------------

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            rendered_image_key: rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def set_mlp(self, mlp: nn.Module):
        self.mlp=mlp

    @staticmethod
    def render(
            means3D: torch.Tensor,  # xyz
            opacity: torch.Tensor,
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],
            features: Optional[torch.Tensor],  # shs
            active_sh_degree: int,
            viewpoint_camera,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            colors_precomp: Optional[torch.Tensor] = None,
            cov3D_precomp: Optional[torch.Tensor] = None,
    ):
        if colors_precomp is not None:
            assert features is None
        if cov3D_precomp is not None:
            assert scales is None
            assert rotations is None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            means3D,
            dtype=means3D.dtype,
            requires_grad=True,
            device=means3D.device,
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rasterize_result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=features,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if len(rasterize_result) == 2:
            rendered_image, radii = rasterize_result
            depth_image = None
        else:
            rendered_image, radii, depth_image = rasterize_result

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY),
        }

    def shs_weight_MLP(self, rasterizer, shs, VC, means3D, L):
        """
        가시 가우시안들에 대해 MLP로 sh 가중치 예측. 이후 sh 계수에 가중치를 곱해 반환
        """
        if shs is None:
            return shs

        N, K, C = shs.shape
        device, dtype = shs.device, shs.dtype

        assert C == 3 and K == (L + 1) ** 2

        # 가시 가우시안 인덱스
        if self.mlp is None:
            vis_mask = torch.ones(N, dtype=torch.bool, device=device)
            vis_idx = torch.arange(N, device=device)
        else:
            with torch.no_grad(): vis_mask = rasterizer.markVisible(means3D) # -> (N,) bool 텐서 반환
            vis_idx = torch.where(vis_mask)[0]
            

        means3D_vis = means3D[vis_idx]  # 가시 가우시안 좌표
        shs_vis = shs[vis_idx]  # 가시 가우시안 SH 계수
        
        # sh 가중치(MLP 결과)를 담을 더미 텐서
        sh_weight = torch.zeros(N, L+1, C, device=device, dtype=dtype)

        if isinstance(self.mlp, CigaMLP):
            self.mlp.to_input(VC, means3D_vis)

            sh_weight = self.mlp(
                # 카메라 좌표
                # 거리
                # 방향
            )
        else:
            print("!!!")
        
        # 밴드별 가중치를 계수별 가중치로 변환
        #print("shs_weight_MLP:", sh_weight.shape)  # -> shs_weight_MLP: torch.Size([219439, 4, 3])
        sh_weight = self.band_flatten(sh_weight, L)

        # sh 계수와 가중치 곱 (MLP 출력 형태에 따라 연산 수정 필요)
        # shs shape         = [gaussian 수, 3(RGB), (max_sh_degree+1)^2]
        # sh_weight shape   = [가시 가우시안 수, 3(RGB), max_sh_degree+1]
        shs_out = shs.clone()
        shs_out = shs * sh_weight
        return shs_out

    def band_flatten(self, sh_weight, L):
        """
        sh_weight: [N, L+1, 3]
        반환:      [N, (L+1)^2, 3]
        """
        import torch

        N, B, C = sh_weight.shape
        assert C == 3 and B == L + 1, f"got {sh_weight.shape}, expected [N,{L+1},3]"

        # 각 밴드의 계수 개수: 2l+1  (0차:1, 1차:3, ..., L차:2L+1)
        band_sizes = torch.tensor([2 * l + 1 for l in range(L + 1)],
                                device=sh_weight.device)

        # 길이 K=((L+1)^2) 벡터: 각 계수 k가 어느 밴드에 속하는지 (0..L)
        band_of_coeff = torch.repeat_interleave(
            torch.arange(L + 1, device=sh_weight.device),
            band_sizes
        )  # [K]

        # 밴드 축(dim=1) 기준으로 인덱싱 → [N, K, 3]
        weights_coeff = sh_weight.index_select(dim=1, index=band_of_coeff)
        
        return weights_coeff

        
        