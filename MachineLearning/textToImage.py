from PIL import Image, ImageDraw
text = "WHO ASKED?!?!?!?!?"
img = Image.new('RGB', (len(text)*10, 30), color=(73, 109, 137))

d = ImageDraw.Draw(img)
d.text((10, 10), text, fill=(255, 255, 0))

img.save('pil_text.png')
img = Image.open('pil_text.png')
img.show()
