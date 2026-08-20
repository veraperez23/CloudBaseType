import torch
import random
import io
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

class AllSky:
    def __init__(self):
        self.rotaciones = [0, 90, 180, 270]
    
    def __call__(self, x):

        angulo_aleatorio = np.random.choice(self.rotaciones, p=[0.25, 0.25, 0.25, 0.25])

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.Lambda(lambda img: img.rotate(angulo_aleatorio)),
            transforms.Resize((512, 512), interpolation=Image.BILINEAR),  
            transforms.ToTensor()
        ])

        return transform(x)


class PhysicsAwareAugmentation:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.rotations = [0, 90, 180, 270]

    def _apply_hot_pixels(self, t):
        """Simulates thermal noise (Dark Current) common in long exposures."""
        if random.random() > self.cfg.low_light.hot_pixels_prob:
            return t
        
        # Create a mask for salt noise (white pixels)
        mask = torch.rand_like(t) < self.cfg.low_light.hot_pixels_ratio
        # Set pixel value to max (1.0) where mask is True
        t[mask] = 1.0 
        return t

    def _apply_night_noise(self, t):
        """
        Injects heteroscedastic noise: 
        1. Poisson (Signal-dependent): Dominates in signal areas (stars).
        2. Gaussian (Signal-independent): Dominates in shadow areas (read noise).
        """
        # Poisson Component (Shot Noise)
        scale = random.uniform(*self.cfg.low_light.noise_poisson_scale)
        noise_p = torch.randn_like(t) * t 
        
        # Gaussian Component (Read/Thermal Noise)
        sigma = random.uniform(*self.cfg.low_light.noise_gaussian_sigma)
        noise_g = torch.randn_like(t) * sigma
        
        total_noise = (noise_p * scale) + noise_g
        return torch.clamp(t + total_noise, 0., 1.)

    def _apply_blooming(self, img):
        """Simulates sensor blooming: charge overflow from saturated pixels."""
        # Only apply soft glow to very bright areas
        # 1. Extract highlights
        grayscale = img.convert("L")
        # Threshold: only consider pixels > 200 brightness
        mask = grayscale.point(lambda p: 255 if p > 220 else 0)
        
        # 2. Blur the highlights
        blur_radius = random.uniform(*self.cfg.high_light.bloom_radius)
        glow = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # 3. Add glow back to original image
        # We convert glow to RGB to blend
        glow_layer = glow.convert("RGB")
        # Screen blending or simple addition
        return Image.composite(ImageEnhance.Brightness(img).enhance(1.2), img, glow)

    def _quantize_to_8bit(self, t):
        """
        Discretizes continuous tensor values to 8-bit integers [0, 255].
        Crucial for domain alignment with standard image formats.
        """
        return (t * 255.0).round().clamp(0, 255) / 255.0

    def _apply_jpeg(self, img, quality_range):
        """Introduces block artifacts (DCT quantization error)."""
        quality = random.randint(*quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def __call__(self, img):
        # 1. Geometric Invariance (Time-independent)
        if random.random() < self.cfg.geom.prob_flip:
            img = ImageOps.mirror(img)
        if random.random() < self.cfg.geom.prob_flip:
            img = ImageOps.flip(img)
        angle = random.choice(self.rotations)
        img = img.rotate(angle)
        
        # Resize logic (Bicubic for sharpness preservation)
        img = img.resize((512, 512), resample=Image.BICUBIC)
        
        # PHYSICAL BRANCHING
        p_regime = random.random()
        
        # LOW-LIGHT --> SNR degradation, Integration Blur, Underexposure
        if p_regime < self.cfg.low_light.prob_apply:

            sigma = random.uniform(*self.cfg.low_light.blur_sigma)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            exp_factor = random.uniform(*self.cfg.low_light.exposure_factor)
            img = ImageEnhance.Brightness(img).enhance(exp_factor)
            t = transforms.ToTensor()(img)
            t = self._apply_night_noise(t)
            t = self._apply_hot_pixels(t)

        # HIGH-LIGHT --> High SNR, Saturation, Clipping, Sharpness
        else:

            exp_factor = random.uniform(*self.cfg.high_light.exposure_factor)
            sat_factor = random.uniform(*self.cfg.high_light.saturation_factor)
            con_factor = random.uniform(*self.cfg.high_light.contrast_factor)

            img = ImageEnhance.Brightness(img).enhance(exp_factor)
            img = ImageEnhance.Color(img).enhance(sat_factor)
            img = ImageEnhance.Contrast(img).enhance(con_factor)
            
            # B. Sensor Blooming (Optional simulation of bright sources)
            if random.random() < 0.3:
                img = self._apply_blooming(img)

            # C. Compression Artifacts
            # Day images have high freq gradients (blue sky) where JPEG blocks are visible.
            img = self._apply_jpeg(img, self.cfg.high_light.jpeg_quality)
            
            t = transforms.ToTensor()(img)

        t = self._quantize_to_8bit(t)
        
        return t