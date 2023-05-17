from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# png_prediction_Israel = Image.open('prediction_Israel.png')
# resized_image = png_prediction_Israel.resize((1440, 800))
# resized_image.save('prediction_Israel.tif', format='TIFF')
# tif_prediction_Israel = Image.open('prediction_Israel.tif')
#
# png_transmissionrates_Israel = Image.open('transmissionrates_Israel.png')
# resized_image = png_transmissionrates_Israel.resize((1440, 800))
# resized_image.save('transmissionrates_Israel.tif', format='TIFF')
# tif_transmissionrates_Israel = Image.open('transmissionrates_Israel.tif')
#
# image1 = Image.open('prediction_Israel.tif')
# image2 = Image.open('transmissionrates_Israel.tif')
# width1, height1 = image1.size
# width2, height2 = image2.size
# combined_width = width1 + width2
# combined_height = max(height1, height2)
# combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
# combined_image.paste(image1, (0, 0))
# combined_image.paste(image2, (width1, 0))
# draw = ImageDraw.Draw(combined_image)
# font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 36)
# draw.text((100, height1-100), "A", fill="black", font=font)
# draw.text((width1, height1-100), "B", fill="black", font=font)
# combined_image.save('combined_Israel.tif')
# combined_image.save('combined_Israel.png')


png_comparison_qua_Israel = Image.open('comparison_quarantined_Israel.png')
resized_image = png_comparison_qua_Israel.resize((1440, 800))
resized_image.save('comparison_quarantined_Israel.tif', format='TIFF')
tif_comparison_qua_Israel = Image.open('comparison_quarantined_Israel.tif')

png_comparison_death_Israel = Image.open('comparison_deaths_Israel.png')
resized_image = png_comparison_death_Israel.resize((1440, 800))
resized_image.save('comparison_deaths_Israel.tif', format='TIFF')
tif_comparison_death_Israel = Image.open('comparison_deaths_Israel.tif')

image1 = Image.open('comparison_quarantined_Israel.tif')
image2 = Image.open('comparison_deaths_Israel.tif')
width1, height1 = image1.size
width2, height2 = image2.size
combined_width = width1 + width2
combined_height = max(height1, height2)
combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
combined_image.paste(image1, (0, 0))
combined_image.paste(image2, (width1, 0))
draw = ImageDraw.Draw(combined_image)
font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 36)
draw.text((100, height1-100), "A", fill="black", font=font)
draw.text((width1 + 50, height1-100), "B", fill="black", font=font)
combined_image.save('combined_comparison_Israel.tif')
combined_image.save('combined_comparison_Israel.png')