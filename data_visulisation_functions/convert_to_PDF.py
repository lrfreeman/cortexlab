import glob
from fpdf import FPDF

images = glob.glob("/Users/laurence/Desktop/rasters/lock_to_reward/*.png")
pdf = FPDF()

# imagelist is the list with all image filenames
for image in images:
    pdf.add_page()
    pdf.image(image,0,0,210,297)
pdf.output("reward_rasters.pdf", "F")
