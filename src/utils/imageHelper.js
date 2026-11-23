/**
 * Utility to convert an Image/SVG Data URL into a real JPEG File object
 * so example memes can be submitted directly to the FastAPI multipart endpoint.
 */

export function svgDataUrlToJpegFile(dataUrl, filename = 'example_meme.jpg') {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width || 400;
      canvas.height = img.height || 400;
      const ctx = canvas.getContext('2d');
      // Fill white background for JPEG
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File([blob], filename, { type: 'image/jpeg' });
            resolve(file);
          } else {
            reject(new Error('Canvas conversion to blob failed'));
          }
        },
        'image/jpeg',
        0.95
      );
    };
    img.onerror = (e) => reject(new Error('Failed to load SVG data URL for canvas rendering'));
    img.src = dataUrl;
  });
}
