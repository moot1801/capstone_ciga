
import math
import torch
from typing import Optional, Dict

# from .renderer import *
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh

from internal.models.sh_core import compute_distance, compute_nadir_angle, d_to_u, phi_to_v, apply_adaptive_sh_weights, GateLUT, GateMLP




class CigaRenderer(Renderer):

    # def __init__(self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False):

    #     super().__init__()
    #     self.compute_cov3D_python = compute_cov3D_python
    #     self.convert_SHs_python = convert_SHs_python


    def __init__(self, compute_cov3D_python=False, convert_SHs_python=False):
        super().__init__()
        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

        
        self.use_adaptive_sh = True
        self.MLP_mode = "visible"  # ["visible", "all"]
        self.d_low = 0.2
        self.d_high = 80.0  #5.0
        self.d0 = 5.0  #1.0
        self.warp_weight = 0.5

        self.MLP = GateMLP(in_dim=2, L_max=3, hidden=32)
        self.LUT = GateLUT(L_max=3, B_d=32, B_phi=32, ema=0.9, device="cuda")
        self.lut_update_every = 500
        self._step = 0
        u_lin = torch.linspace(0, 1, self.LUT.B_d)
        v_lin = torch.linspace(0, 1, self.LUT.B_phi)
        grid_v, grid_u = torch.meshgrid(v_lin, u_lin, indexing="ij")
        self.register_buffer("grid_u", grid_u)
        self.register_buffer("grid_v", grid_v)



    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            render_types: list = None,

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

        #shs = self.shs_weight_MLP(rasterizer, shs, means3D, L=pc.max_sh_degree)
        
        
        if self.use_adaptive_sh and shs is not None:
                    shs = self.apply_adaptive_weights(
                        rasterizer, shs, means3D, 
                        viewpoint_camera.camera_center,
                        L=pc.max_sh_degree
                    )


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

        
        extra = {}
        if self.use_adaptive_sh and shs is not None:
            shs, sh_weights, distances, nadir_angles = self.apply_adaptive_weights(rasterizer, shs, means3D, viewpoint_camera.camera_center,
        L=pc.active_sh_degree,   # 혹은 shs에서 유추하도록 apply 내부를 바꾸면 이 줄은 L 생략
        return_aux=True)
            extra.update(dict(sh_weights=sh_weights, distances=distances, nadir_angles=nadir_angles))
        


        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.

        # They will be excluded from value updates used in the splitting criteria.

        return {

            rendered_image_key: rendered_image,

            "viewspace_points": screenspace_points,

            "visibility_filter": radii > 0,

            "radii": radii,
            **extra,

        }
    

    def apply_adaptive_weights(self, rasterizer, shs, means3D, camera_center, L, return_aux=False):
        self.MLP = self.MLP.to(means3D.device)
        self.LUT = self.LUT.to(means3D.device)

        if shs is None:
            return shs
        
        N, C, K = shs.shape
        L = int(math.isqrt(K)) - 1
        device, dtype = shs.device, shs.dtype
        assert C == 3 and K == (L + 1) ** 2
        

        if self.MLP_mode == 'visible':
            with torch.no_grad():
                vis_mask = rasterizer.markVisible(means3D).to(means3D.device)
            vis_idx = torch.where(vis_mask)[0]
            
        else:
            vis_mask = torch.ones(N, dtype=torch.bool, device=device)
            vis_idx = torch.arange(N, device=device)
        
        means3D_vis = means3D[vis_idx]
        

        distances = compute_distance(camera_center, means3D_vis)
        nadir_angles = compute_nadir_angle(camera_center, means3D_vis)
        

        u = d_to_u(distances, self.d_low, self.d_high, self.d0, self.warp_weight)
        v = phi_to_v(nadir_angles)
        
 
        if self.training:
            uv = torch.stack([u, v], dim=-1)
            sh_weights_vis = self.MLP(uv)
            self._step += 1
            if self._step % self.lut_update_every == 0:
                with torch.no_grad():
                    self.LUT.update_from_mlp(self.MLP, self.grid_u, self.grid_v)
                
            
        else:
            sh_weights_vis = self.LUT.lookup(u, v)
        

        sh_weights = torch.ones(N, L + 1, device=device, dtype=dtype)
        sh_weights[vis_idx] = sh_weights_vis
        
  
        shs_out = apply_adaptive_sh_weights(shs, sh_weights, L)
        

        if return_aux:
            dist_all = torch.zeros(N, device=device, dtype=dtype)
            phi_all  = torch.zeros(N, device=device, dtype=dtype)
            dist_all[vis_idx] = distances
            phi_all[vis_idx] = nadir_angles
            return shs_out, sh_weights, dist_all, phi_all
        
        return shs_out
    
    



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



    # def shs_weight_MLP(self, rasterizer, shs, means3D, L):

    #     """

    #     가시 가우시안들에 대해 MLP로 sh 가중치 예측. 이후 sh 계수에 가중치를 곱해 반환

    #     """

    #     if shs is None:

    #         return shs



    #     N, C, K = shs.shape

    #     device, dtype = shs.device, shs.dtype



    #     assert C == 3 and K == (L + 1) ** 2



    #     # 가시 가우시안 인덱스

    #     if self.MLP_mode == 'visible':

    #         with torch.no_grad(): vis_mask = rasterizer.markVisible(means3D) # -> (N,) bool 텐서 반환

    #         vis_idx = torch.where(vis_mask)[0]

    #     else:

    #         vis_mask = torch.ones(N, dtype=torch.bool, device=device)

    #         vis_idx = torch.arange(N, device=device)



    #     means3D_vis = means3D[vis_idx]  # 가시 가우시안 좌표

    #     shs_vis = shs[vis_idx]  # 가시 가우시안 SH 계수

        

    #     # sh 가중치(MLP 결과)를 담을 더미 텐서

    #     sh_weight = torch.zeros(sum(vis_mask), C, L + 1, device=device, dtype=dtype)



    #     #Ciga MLP

    #     # for i, xyz in enumerate(means3D_vis):

    #     #     sh_weight[i] = self.MLP(

    #     #         xyz,

    #     #         # 카메라-가우시안 거리

    #     #         # 카메라 방향

    #     #     )

    #     sh_weight = self.MLP(

    #         means3D_vis,

    #         # 카메라-가우시안 거리,

    #         # 카메라 방향

    #         )  

        

    #     # 밴드별 가중치를 계수별 가중치로 변환

    #     sh_weight = self.band_flatten(sh_weight, L)



    #     # sh 계수와 가중치 곱 (MLP 출력 형태에 따라 연산 수정 필요)

    #     # shs shape         = [gaussian 수, 3(RGB), (max_sh_degree+1)^2]

    #     # sh_weight shape   = [가시 가우시안 수, 3(RGB), max_sh_degree+1]

    #     shs_out = shs.clone()

    #     shs_out[vis_idx] = shs[vis_idx] * sh_weight

    #     return shs_out



    # def band_flatten(self, sh_weight, L):

    #     """

    #     sh_weight: [N, 3, L+1]  

    #     반환:      [N, 3, (L+1)^2] 

    #     """

    #     N, C, B = sh_weight.shape

    #     assert C == 3 and B == L + 1

    #     K = (L + 1) ** 2



    #     # 각 밴드의 계수 개수: 2l+1

    #     # ex: 0차 : 1개     / 1차 : 3개     / 2차 : 5개     / ...   / L차 : 2L+1개

    #     band_sizes = torch.tensor([2*l + 1 for l in range(L + 1)],

    #                             device=sh_weight.device)



    #     # 길이 K 벡터: 각 계수 k가 어느 밴드에 속하는지 표시

    #     band_of_coeff = torch.repeat_interleave(

    #         torch.arange(L + 1, device=sh_weight.device), band_sizes

    #     )  # shape: [K], 값 범위 0..L



    #     # 밴드축(마지막 축)에서 인덱스 선택으로 복제 확장

    #     weights_coeff = sh_weight.index_select(dim=2, index=band_of_coeff)  # [N,3,K]

    #     return weights_coeff