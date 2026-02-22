from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
import os

def apply_cyberpunk_filter(image_path, output_path="cyberpunk_image.png"):
    try:
        # Load image and ensure RGB mode
        img = Image.open(image_path).convert("RGB")
        
        # 1. Enhance Color Saturation heavily
        enhancer_color = ImageEnhance.Color(img)
        img_colored = enhancer_color.enhance(2.5)  # Boost colors
        
        # 2. Enhance Contrast
        enhancer_contrast = ImageEnhance.Contrast(img_colored)
        img_contrasted = enhancer_contrast.enhance(1.5) # Increase contrast
        
        # 3. Extract edges using a filter
        edges = img_contrasted.filter(ImageFilter.FIND_EDGES)
        
        # 4. Colorize the edges to a bright neon cyan/blue
        # Convert edges to grayscale first to act as a proper mask for colorize
        edges_l = edges.convert("L")
        # Enhance edge intensity
        edges_enhanced = ImageEnhance.Contrast(edges_l).enhance(3.0)
        edges_neon = ImageOps.colorize(edges_enhanced, black="black", white="#00FFFF")
        
        # 5. Blend the neon edges back onto the saturated image using Screen
        final_img = ImageChops.screen(img_contrasted, edges_neon)
        
        # Save output
        final_img.save(output_path)
        print(f"Cyberpunk filtered image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Cyberpunk Neon Filter Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_cyberpunk{ext}"
        apply_cyberpunk_filter(image_path, output_file)
