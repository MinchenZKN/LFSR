from skimage import measure

def PSNR_SSIM_CROSS(Ref_h,Ref_v,Res_h,Res_v):
    views = Ref_h.shape[0]
    PSNR = 0
    SSIM = 0

    for k in range(0,views):
        PSNR_h = measure.compare_psnr(Ref_h[k,:,:,:], Res_h[k,:,:,:], data_range=255, dynamic_range=None)
        SSIM_h = measure.compare_ssim(Ref_h[k,:,:,:], Res_h[k,:,:,:], data_range=255, multichannel=True)

        PSNR_v = measure.compare_psnr(Ref_v[k,:,:,:], Res_v[k,:,:,:], data_range=255, dynamic_range=None)
        SSIM_v = measure.compare_ssim(Ref_v[k,:,:,:], Res_v[k,:,:,:], data_range=255, multichannel=True)

        PSNR = PSNR + PSNR_h + PSNR_v
        SSIM = SSIM + SSIM_h + SSIM_v

    return PSNR/(views*2-1), SSIM/(views*2-1)