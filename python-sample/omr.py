# This is just a sample program to show you how to do
# basic image operations using python and the Pillow library.
#
# By Eriya Terada, based on earlier code by Stefan Lee,
#    lightly modified by David Crandall, 2020

# Import the Image and ImageFilter classes from PIL (Pillow)

from PIL import Image, ImageFilter, ImageDraw, ImageFont
import random
import numpy as np
import sys


# Step 3 Convert image to gray scale
def grayscale_pad(image, padding_size):
    im = Image.open(image).convert("L")
    im_width = im.width
    im_height = im.height
    new_width = (2 * padding_size) + im_width
    new_height = (2 * padding_size) + im_height

    # Create a new blank grayscale image with padding
    gray_im = Image.new("L", (new_width, new_height), color=255)

    # Loop over the new image with padding
    for x in range(new_width):
        for y in range(new_height):
            # fill in areas that are not padding
            if x > padding_size and x < new_width - padding_size:
                if y > padding_size and y < new_height - padding_size:
                    # convert the original image to grayscale
                    l_value = im.getpixel((x - padding_size, y - padding_size))
                    gray_im.putpixel((x, y), l_value)

    # Save the image
    gray_im.save("gray.png")
    return gray_im

# Step 4 Convolution with separable kernel
def convolve(image, hx, hy):

    im_width = image.width
    im_height = image.height
    hx_len = len(hx)
    hy_len = len(hy)
    image=np.array(image).astype(np.uint8)
    

    new_image = np.zeros(image.shape)
    vertimage = np.zeros(image.shape)
 
    # convolve vertically
    for x in range(im_height-hy_len+1):
        for y in range(im_width):
            row_sum=0
            col_sum=0
            for v in range(hy_len):
                row_sum+=image[x+v][y]*hy[v]
            vertimage[x][y]=row_sum
     
    # convolve horizontally
    img = Image.fromarray(np.uint8(vertimage * 255))
    for x in range(im_height):
        for y in range(im_width-hx_len+1):
            row_sum=0
            col_sum=0
            for h in range(hx_len):
                col_sum+=vertimage[x][y+h]*hx[h]
            new_image[x][y]=col_sum    
          

    img = Image.fromarray(np.uint8(new_image * 255))
#    img.show()
    return img
                
# Canny edge detection
def sobel_edge_detection(gray_img):
    
    gray_img=np.array(gray_img).astype(np.uint8)
    # Sobels filter 
    v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
    h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  
    
    print(gray_img.shape)
    
    im_height, im_width = gray_img.shape
  
    new_image_h = np.zeros(gray_img.shape)
    new_image_v = np.zeros(gray_img.shape)
    new_image = np.zeros(gray_img.shape)
    
    for i in range(0, im_height-3+1):
        for j in range(0, im_width-3+1):
            horizontalGrad=0
            verticalGrad=0
            for x in range(h.shape[0]):
                for y in range(h.shape[1]):
                    horizontalGrad+=h[x][y]*gray_img[i+x,j+y]
                    
            new_image_h[i, j] = abs(horizontalGrad)
            
            for x in range(v.shape[0]):
                for y in range(v.shape[1]):
                    verticalGrad+=v[x][y]*gray_img[i+x,j+y]
                    
            new_image_v[i, j] = abs(verticalGrad)

            # Edge Magnitude
            edge_mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            new_image[i, j] = edge_mag

    img = Image.fromarray(np.uint8(new_image * 255))
    img.show()
    
    # Create binary edge map
    new_image[new_image!= 0.0]=1
    new_image[new_image== 0.0]=0
    print(new_image.shape)
    return new_image

def get_region_colors(im, t_height, t_width, coordinate):
    # coordinate is the x,y value of where the region starts in the image
    # region_colors is the same size as the template
    region_colors = []
    for i in range(coordinate[0], coordinate[0]+t_height):
        row = []
        for j in range(coordinate[1], coordinate[1]+t_width):
            row.append(im.getpixel((j, i)))
        region_colors.append(row)
    return region_colors

def compareImages(region, template):
    # takes 2 matrices with the color values
    # region and template are the same size
    t_height = len(template)
    t_width = len(template[0])
    total_score = 0

    for i in range(t_height):
        for j in range(t_width):
            region_pixel = region[i][j]
            t_pixel = template[i][j]
            # changed similarity function to use 255 instead of 1 since grayscale values are from 0-255
            pixel_similarity = (region_pixel * t_pixel) + (255-region_pixel) * (255-t_pixel)
            total_score += pixel_similarity
    return total_score

'''
Function to calculate hamming distance i.e. step 5 in the assignment
'''
def hammingDist(im, t_im, combine, color, text_file_list, symbol_type, p, dist):
    im_width = im.width
    im_height = im.height
    t_width = t_im.width
    t_height = t_im.height

    # get the template and it's score to compare with image regions later on
    t_region = get_region_colors(t_im, t_height, t_width, (0,0))
    perfect_score = compareImages(t_region, t_region)

    #t_found = Image.new("L", (im_width, im_height), color=255)

    combine = combine.copy().convert("RGB")
    d = {}
    # loop through the image
    for i in range(im_height-t_height):
        for j in range(im_width-t_width):
            # get image region
            im_region = get_region_colors(im, t_height, t_width, (i, j))
            # score the region
            region_score = compareImages(im_region, t_region)
            # compare the image region score to the template score
            if region_score >= (0.87 * perfect_score):
                max_val = region_score
                it_val = (i,j)
                for y in range(3):
                    for z in range(3):
                        if (i-y,j-z) in d:
                            if d[(i-y,j-z)] >= region_score:
                                max_val = region_score
                                it_val = (i-y,j-z)
                            else: 
                                del d[(i-y,j-z)]
                        elif (i-y,j+z) in d:
                            if d[(i-y,j+z)] >= region_score:
                                max_val = region_score
                                it_val = (i-y,j+z)
                            else: 
                                del d[(i-y,j+z)]
                d[it_val] = max_val
                
    for k,v in d.items(): 
        i,j = k
        region_score = v           
        draw = ImageDraw.Draw(combine)
        top_left = (j,i)
        bottom_right = (j + t_width, i + t_height)
        #draw.rectangle(((100, 100), (200, 200)), (0, 255, 0))
        draw.rectangle((top_left, bottom_right), fill=None, outline = color,width=2)
        pitch = '_'
        
        if symbol_type == 'filled_note':
            for q in range(int(dist/2)):
                if q+i in p:
                    pitch = p[q+i]
                elif i-q in p:
                    pitch = p[i-q]

            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuLGCSansMono.ttf")
            # font = ImageFont.truetype("/usr/share/fonts/msttcorefonts/arial.ttf")  load_default()
            draw.text((j-10, i-2),pitch,(255,0,0),font=font)
        
        text_file_list.append([j, i, t_height, t_width, symbol_type, pitch, float(round((region_score/perfect_score*100), 2))])
#    combine.save("step5.png")
    return combine, text_file_list

# Step 6: Template matching using convolution
def template_matching(image, template):
    m=template.shape[0]
    n=template.shape[1]

    F=np.zeros((image.shape))
    D=np.zeros((image.shape))
    
#    X=np.array(image)
# 
#    X[X==0]=np.inf
#    X[X==1]=0
    
    # Find the coordinates of edges
    v,w=np.where(image!=0)
    loc=np.stack((v,w),axis=1)
    
    # Find coordinates of whole image
    v1,w1=np.where(image==0)
    loc1=np.stack((v1,w1),axis=1)
    loc2=np.vstack((loc,loc1))
 
    # Calculate D matrix which stores the distance of each pixel from its nearest edge pixel
    temp=np.zeros(loc.shape[0])
    for i in range(loc2.shape[0]):
        temp=np.sqrt((loc2[i][0]-loc[:,0])**2+(loc2[i][1]-loc[:,1])**2)
        D[loc2[i][0],loc2[i][1]]=np.min(temp)

    
    img = Image.open(im_name)
    draw = ImageDraw.Draw(img)
    
    sum=0
    for k in range(0,m):
        for l in range(0,n):
            sum+=(template[k][l])*(template[k][l])
    score=sum
    
    max_D=np.max(D)
  
    # Calculate template scoring
    for i in range(0,image.shape[0]-m+1):
        for j in range(0,image.shape[1]-n+1):
            sum=0
            for k in range(0,m):
                for l in range(0,n):                 
                    sum+=((template[k][l])*((max_D-D[i+k][j+l])/max_D))
            F[i][j]=sum
            if sum>=0.95*score:
                draw.rectangle(((j,i), (j+n,i+m)), fill=None,outline="red")

    img.save("output-6.png")

def hough_line(edge):
    thetas = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))

    rho_range = round(math.sqrt(edge.shape[0]*2 + edge.shape[1]*2))
    accumulator = np.zeros((2 * rho_range, len(theta)), dtype=np.uint8)

    edge_pixels = np.where(edge == 1)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    for p in range(len(coordinates)):
        for theta in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[theta] + coordinates[p][0] * sin[theta]))
            accumulator[rho, t] += 1 
    #print(np.max(accumulator))
    return accumulator

def hough(image):
    
#    im = image.load()
#    im_h, im_w = image.size
#    th_val, r_val = 500, 1200
#    hough_im = Image.new("L", (th_val, r_val), 255)
#    him = hough_im.load()
#    rho = {}
#    rmax = hypot(im_h, im_w)
#    dr = rmax / int(r_val/2)
#    dth = pi / th_val
#    for x in range(im_h):
#        for y in range(im_w):
#            if im[x, y] != 255:
#                for m in range(th_val):
#                    th = dth * m
#                    r = x*cos(th) + y*sin(th)
#                    n = int(r_val/2) + int(r/dr+0.5)
#                    him[m, n] -= 1
    
    dist = 0
    img = image.convert('L')  #conversion to gray scale
    bw = img.point(lambda x: 0 if x<128 else 255, '1')
    img_bin = np.array(bw).astype(np.uint8)
    x, y = img_bin.shape
    d = {}
    for i in range(0,x):
        d[i] = 0
        for j in range(y):
            if img_bin[i][j]==0:
                d[i] +=1
    l = [k for k,v in d.items() if v > y/2]
    for i in range(0,len(l)-1):
        if l[i]+1 != l[i+1]:
            if dist == 0:
                dist = l[i+1]-l[i]
            elif dist == l[i+1]-l[i]:
                break
            
    lines = [l[0]]
    p = l[0]
    for i in range(1,len(l)):
        if l[i] - p > dist*2:
            lines.append(l[i])
        p = l[i]
    
    return dist, lines
    
def rescale(template,dist):
    temp = Image.open(template).convert("L")
    factor = dist/temp.height
    temp = temp.resize((int(temp.width * factor), int(temp.height * factor)))
    return temp

def pitch(lines,dist):
    p = {}
    j = 1
    for i in lines:
        if j%2 ==0:
            p[i-dist*1.5] = 'D'
            p[i-dist] = 'C'
            p[i-dist*0.5] = 'B'
            p[i] = 'A'
            p[i+dist*0.5] = 'G'
            p[i+dist] = 'F'
            p[i+dist*1.5] = 'E'
            p[i+dist*2] = 'D'
            p[i+dist*2.5] = 'C'
            p[i+dist*3] = 'B'
            p[i+dist*3.5] = 'G'
            p[i+dist*4] = 'F'
            p[i+dist*4.5] = 'E'
        else:
            p[i-dist*0.5] = 'G'
            p[i] = 'F'
            p[i+dist*0.5] = 'E'
            p[i+dist] = 'D'
            p[i+dist*1.5] = 'C'
            p[i+dist*2] = 'B'
            p[i+dist*2.5] = 'A'
            p[i+dist*3] = 'G'
            p[i+dist*3.5] = 'F'
            p[i+dist*4] = 'E'
            p[i+dist*4.5] = 'D'
            p[i+dist*5] = 'B'
        j += 1
    return p    

if __name__ == '__main__':
    music_file = sys.argv[1]
    im_name = "../test-images/" + music_file
    template1 = "../test-images/template1.png"
    template2 = "../test-images/template2.png"
    template3 = "../test-images/template3.png"
    template4 = "../test-images/template4.png"
    template5 = "../test-images/template5.png"
        
    image = Image.open(im_name)
#    finding the scale of the template
    dist, lines = hough(image)
    temp1 = rescale(template1,dist)
    temp2 = rescale(template2,dist*3)
    temp3 = rescale(template3,dist*2.5)
    temp4 = rescale(template4,dist*3)
    temp5 = rescale(template5,dist*8)
    
    gray_im = image.convert("L")
#    temp1 = Image.open(template1).convert("L")
#    temp2 = Image.open(template2).convert("L")
#    temp3 = Image.open(template3).convert("L")    
    
#    hx=[1,2,1]
#    hy=[1,2,1]
#    image=convolve(gray_im, hx, hy)
    
#    edge1=sobel_edge_detection(gray_im)
#    edge2=sobel_edge_detection(temp1)
    
#    template_matching(edge1,edge2)
    
    result_list = []
    l =[]
    
    p = pitch(lines,dist)

    result1, result_list = hammingDist(gray_im, temp1, gray_im, "red", result_list, "filled_note", p, dist)
    result2, result_list = hammingDist(gray_im, temp2, result1, "green", result_list, "eighth_rest", p, dist)
    result3, result_list = hammingDist(gray_im, temp3, result2, "blue", result_list, "quarter_rest", p, dist)
    result4, l = hammingDist(gray_im, temp4, result3, "yellow", l, "quarter_rest", p, dist)
    result5, l = hammingDist(gray_im, temp5, result4, "pink", l, "quarter_rest", p, dist)
    text_list = result_list
    np.savetxt("detected.txt", text_list, fmt="%s") # Saving the results in a txt file
    result5.save("detected.png")
