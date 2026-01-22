"""
Unified image I/O module for pyalicevisionlib.

Provides consistent image loading and saving with fallback support:
1. OpenImageIO (oiio) - Best for EXR, HDR, and professional formats
2. OpenCV (cv2) - Good general support, fast
3. Matplotlib (plt) / PIL - Fallback for basic formats

All functions use RGB/RGBA channel order (not BGR).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

# Try importing image libraries in priority order
try:
    import OpenImageIO as oiio
    HAS_OIIO = True
except ImportError:
    HAS_OIIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_image(
    image_path: Union[str, Path],
    mode: str = 'rgb',
    dtype: str = 'uint8'
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        mode: Color mode - 'rgb', 'rgba', 'gray', or 'unchanged'
        dtype: Output dtype - 'uint8' (0-255) or 'float32' (0-1)
        
    Returns:
        Image as numpy array in RGB(A) order, shape (H, W) or (H, W, C)
        
    Raises:
        RuntimeError: If image cannot be loaded
        ValueError: If invalid mode or dtype specified
    """
    path = Path(image_path)
    
    if mode not in ('rgb', 'rgba', 'gray', 'unchanged'):
        raise ValueError(f"Invalid mode: {mode}. Use 'rgb', 'rgba', 'gray', or 'unchanged'")
    if dtype not in ('uint8', 'float32'):
        raise ValueError(f"Invalid dtype: {dtype}. Use 'uint8' or 'float32'")
    
    img = None
    
    # Try OpenImageIO first (best for EXR, HDR)
    if HAS_OIIO:
        img = _load_with_oiio(path, mode)
    
    # Fallback to OpenCV
    if img is None and HAS_CV2:
        img = _load_with_cv2(path, mode)
    
    # Fallback to PIL
    if img is None and HAS_PIL:
        img = _load_with_pil(path, mode)
    
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    
    # Convert dtype
    if dtype == 'float32':
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
    elif dtype == 'uint8':
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
    
    return img


def _load_with_oiio(path: Path, mode: str) -> Optional[np.ndarray]:
    """Load image using OpenImageIO."""
    try:
        img_buf = oiio.ImageBuf(str(path))
        if not img_buf.initialized or not img_buf.read(0, 0, True, oiio.TypeFloat):
            return None
        
        spec = img_buf.spec()
        h, w, c = spec.height, spec.width, spec.nchannels
        pixels = img_buf.get_pixels(oiio.TypeFloat)
        img = np.array(pixels, dtype=np.float32).reshape(h, w, c)
        
        # Handle HDR/EXR normalization
        if img.max() > 1.0 or path.suffix.upper() in ('.EXR', '.HDR'):
            percentile_99 = np.percentile(img, 99.5)
            if percentile_99 > 0:
                img = img / percentile_99
            img = np.clip(img, 0, 1)
        
        # Convert to requested mode
        if mode == 'gray':
            if c >= 3:
                img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            else:
                img = img[:, :, 0]
        elif mode == 'rgb':
            if c >= 3:
                img = img[:, :, :3]
            elif c == 1:
                img = np.stack([img[:, :, 0]] * 3, axis=-1)
        elif mode == 'rgba':
            if c >= 4:
                img = img[:, :, :4]
            elif c >= 3:
                alpha = np.ones((h, w, 1), dtype=np.float32)
                img = np.concatenate([img[:, :, :3], alpha], axis=-1)
            elif c == 1:
                alpha = np.ones((h, w, 1), dtype=np.float32)
                img = np.concatenate([np.stack([img[:, :, 0]] * 3, axis=-1), alpha], axis=-1)
        # 'unchanged' keeps original
        
        return img
    except Exception:
        return None


def _load_with_cv2(path: Path, mode: str) -> Optional[np.ndarray]:
    """Load image using OpenCV."""
    try:
        flags = cv2.IMREAD_UNCHANGED if mode in ('unchanged', 'rgba') else cv2.IMREAD_COLOR
        if mode == 'gray':
            flags = cv2.IMREAD_GRAYSCALE
        
        img = cv2.imread(str(path), flags)
        if img is None:
            return None
        
        # Convert BGR to RGB
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
        # Handle mode conversions
        if mode == 'rgb' and img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        elif mode == 'rgba' and img.ndim == 3 and img.shape[2] == 3:
            h, w = img.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=img.dtype)
            img = np.concatenate([img, alpha], axis=-1)
        elif mode == 'gray' and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return img
    except Exception:
        return None


def _load_with_pil(path: Path, mode: str) -> Optional[np.ndarray]:
    """Load image using PIL."""
    try:
        with PILImage.open(str(path)) as pil_img:
            if mode == 'rgb':
                pil_img = pil_img.convert('RGB')
            elif mode == 'rgba':
                pil_img = pil_img.convert('RGBA')
            elif mode == 'gray':
                pil_img = pil_img.convert('L')
            # 'unchanged' keeps original mode
            
            return np.array(pil_img)
    except Exception:
        return None


def load_gray(image_path: Union[str, Path], dtype: str = 'uint8') -> np.ndarray:
    """
    Load an image as grayscale.
    
    Args:
        image_path: Path to image file
        dtype: Output dtype - 'uint8' (0-255) or 'float32' (0-1)
        
    Returns:
        Grayscale image as 2D numpy array
    """
    return load_image(image_path, mode='gray', dtype=dtype)


def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """
    Load a mask image as binary.
    
    Args:
        mask_path: Path to mask image (grayscale or first channel used)
        
    Returns:
        Binary mask as 2D boolean numpy array (True = foreground)
    """
    img = load_image(mask_path, mode='unchanged', dtype='uint8')
    
    # Extract single channel if multi-channel
    if img.ndim == 3:
        img = img[:, :, 0]
    
    # Convert to binary
    return img > 127


def load_alpha_as_mask(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load the alpha channel from an RGBA image as a binary mask.
    
    Args:
        image_path: Path to RGBA image (PNG with alpha, EXR, etc.)
        
    Returns:
        Binary mask as 2D boolean numpy array (True = foreground, alpha > 0)
        
    Raises:
        ValueError: If image does not have an alpha channel
    """
    img = load_image(image_path, mode='unchanged', dtype='uint8')
    
    if img.ndim != 3 or img.shape[2] < 4:
        raise ValueError(f"Image does not have alpha channel: {image_path}")
    
    alpha = img[:, :, 3]
    return alpha > 0


def save_image(
    image: np.ndarray,
    image_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array (H, W) or (H, W, C) in RGB(A) order
        image_path: Output path (format determined by extension)
        quality: JPEG quality (0-100), ignored for other formats
        
    Raises:
        RuntimeError: If image cannot be saved
    """
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    success = False
    
    # Try OpenImageIO first
    if HAS_OIIO:
        success = _save_with_oiio(image, path, quality)
    
    # Fallback to OpenCV
    if not success and HAS_CV2:
        success = _save_with_cv2(image, path, quality)
    
    # Fallback to PIL
    if not success and HAS_PIL:
        success = _save_with_pil(image, path, quality)
    
    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def _save_with_oiio(image: np.ndarray, path: Path, quality: int) -> bool:
    """Save image using OpenImageIO."""
    try:
        # Ensure correct shape
        if image.ndim == 2:
            h, w = image.shape
            c = 1
            pixels = image.reshape(h, w, 1)
        else:
            h, w, c = image.shape
            pixels = image
        
        # Determine output type
        if image.dtype in (np.float32, np.float64):
            out_type = oiio.TypeFloat
        elif image.dtype == np.uint16:
            out_type = oiio.TypeUInt16
        else:
            out_type = oiio.TypeUInt8
            if image.dtype != np.uint8:
                pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)
        
        spec = oiio.ImageSpec(w, h, c, out_type)
        
        # Set JPEG quality
        if path.suffix.lower() in ('.jpg', '.jpeg'):
            spec.attribute("compression", f"jpeg:{quality}")
        
        img_buf = oiio.ImageBuf(spec)
        img_buf.set_pixels(oiio.ROI(0, w, 0, h, 0, 1, 0, c), pixels)
        
        return img_buf.write(str(path))
    except Exception:
        return False


def _save_with_cv2(image: np.ndarray, path: Path, quality: int) -> bool:
    """Save image using OpenCV."""
    try:
        img = image.copy()
        
        # Convert to uint8 if needed
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        
        params = []
        if path.suffix.lower() in ('.jpg', '.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif path.suffix.lower() == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        
        return cv2.imwrite(str(path), img, params)
    except Exception:
        return False


def _save_with_pil(image: np.ndarray, path: Path, quality: int) -> bool:
    """Save image using PIL."""
    try:
        img = image.copy()
        
        # Convert to uint8 if needed
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        pil_img = PILImage.fromarray(img)
        
        save_kwargs = {}
        if path.suffix.lower() in ('.jpg', '.jpeg'):
            save_kwargs['quality'] = quality
        
        pil_img.save(str(path), **save_kwargs)
        return True
    except Exception:
        return False


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image data.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        RuntimeError: If dimensions cannot be determined
    """
    path = Path(image_path)
    
    # Try OpenImageIO
    if HAS_OIIO:
        try:
            img_input = oiio.ImageInput.open(str(path))
            if img_input:
                spec = img_input.spec()
                w, h = spec.width, spec.height
                img_input.close()
                return (w, h)
        except Exception:
            pass
    
    # Try PIL (efficient, reads only header)
    if HAS_PIL:
        try:
            with PILImage.open(str(path)) as img:
                return img.size  # (width, height)
        except Exception:
            pass
    
    # Try OpenCV (loads full image, less efficient)
    if HAS_CV2:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
        except Exception:
            pass
    
    raise RuntimeError(f"Cannot determine dimensions for: {path}")


# Module availability info
def get_available_backends() -> dict:
    """
    Get information about available image I/O backends.
    
    Returns:
        Dict with 'oiio', 'cv2', 'pil' keys and bool values
    """
    return {
        'oiio': HAS_OIIO,
        'cv2': HAS_CV2,
        'pil': HAS_PIL
    }
