# Assignment 1
How to run the code
```
cd python-sample
python3 omr.py input_image.png
```
Output:
 
1. detected.png

![Detected Image Output](https://media.github.iu.edu/user/13581/files/38f03880-8cc3-11ea-85b5-129c6c912a2f)

2. detected.txt

The output of the detected templates is stored in the detected.txt file.
Since all the templates provided in the assignment to us are from music1.png, the matching is very efficient in the music1.png file. But for other input images, there are a few nodes that are missed. This could be solved if a generic template is provided which can work on all input images. Also, there is a noticeable variation in the quality of music1.png file and other music files.
We received guidance from Saurabh during the AI hours. He provided a couple of ways to troubleshoot our errors. 

<b> Edits made after previous submission: </b>
Our first submission had a minor error related to the font (OS incompatibility) not being recognized by the SICE server. As per the suggestions by Prof. Crandall and AI Keith, we went ahead with the modifications to the current code. The following edits were made:
1. The font compatible with the server was replaced with the incorrect one. This can be seen in the function 'hammingDist'.

2. We also attempted the extra credit task of detecting the extra symbols (as seen in yellow and pink in the figure above) other than the templates provided. The symbols detected are only visible in the output image and are not printed in the detected.txt file. But this also increased the overall runtime of the code significantly. To run the code without the additional templates just <b>comment</b> these lines:
```
result4, l = hammingDist(gray_im, temp4, result3, "yellow", l, "quarter_rest", p, dist)
result5, l = hammingDist(gray_im, temp5, result4, "pink", l, "quarter_rest", p, dist)

```
and <b>change</b> the below code from 

```result5.save("detected.png")``` to  ```result3.save("detected.png")``` in the main function of omr.py.

3. The output image and the detected text file for music1 are added to the repository. The output image is also added to this readme file.
 
4. We also attempted to implement the Hough transform, but at last, concluded that our method gave more precise output and decided not to use our new Hough implementation.

5. Fixed an error to put an '_' in case of eight_rest and quarter_rest for the pitch value in detected.txt. Previously, it was putting a blank in the file.


## Rescaling template
We are using hough transform to get the staff lines in the image. From the row coordinates of the lines, we can infer the size of a node head, which is the same as the distance between two lines. This value will be used to resize the template.

The image is resized by taking the percentage (factor) by which the template is different from the required size. We then multiply the factor with the height and width of the template and resize it.
```
    factor = dist/temp.height
    temp = temp.resize((int(temp.width * factor), int(temp.height * factor)))
```
## Grayscale and Padding
The following function deals with padding the image and converting it to grayscale. It was decided that images would be padded with a white border since it was the least likely to cause false positives. A white border would make surrounding notes lighter and possibly result in them being ignored in detection. However, adding other colored padding such as black could cause the surrounding pixels be detected as notes when there are none. Mirroring was also considered, but not used because images could contain partial notes that 
would then be mirrored and introduce false positives. 
```
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
```

## Convolution with 2D Kernel

We implemented convolution with a 3x3 Sobel operator to do edge detection. The edge map created was used in step 6. The code for Sobel edge detection is as below:
```
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
```

## Convolution with a Separable Kernel
A function is implemented to convolve an image when provided with a separable kernel hx and hy. To do so, we first perform the convolution of the image and the vertical vector and then convolve the image again with a horizontal vector. This leads to the convolution of the image with the kernel. The function to convolve an image with two separable kernels is done as follows:-
```
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
```

## Hamming distance for template matching

As mentioned in step 5, we find the hamming distance according to the formula provided in the assignment. Since the images are non-binary with grayscale values varying from 0-255, we adjusted the similarity function to fit this new range of values. The input to this section is the output of Hough tranform. First, we compare the two images in the compare images function. Then, the formula provided is implemented in the 'template_matching' function. After calculating the Hamming distance, we provide scoring for the matched templates in the original image. If the match is found with a threshold of 0.87 for that region, we go ahead with drawing the rectangle over the matched region. Every template is colored differently as was required in the output. The input to one of the outputs i.e. detected.txt is also fed in the hammingDist function. 
The function to calculate the hamming distance, template matching and scoring the image is as follows:-
```
def hammingDist(im, t_im, combine, color, text_file_list, symbol_type):
    im_width = im.width
    im_height = im.height
    t_width = t_im.width
    t_height = t_im.height

    t_region = get_region_colors(t_im, t_height, t_width, (0,0))
    perfect_score = compareImages(t_region, t_region)

    #t_found = Image.new("L", (im_width, im_height), color=255)

    combine = combine.copy().convert("RGB")

    #print("im_width: "+str(im_width) + " im_height: " + str(im_height))
    #print("hammingDist: t_width: " + str(t_width) + " t_height: " + str(t_height))
    
    # loop through the image
    for i in range(im_height-t_height):
        for j in range(im_width-t_width):
            # get image region
            im_region = get_region_colors(im, t_height, t_width, (i, j))
            # score the region
            region_score = compareImages(im_region, t_region)
            # compare the image region score to the template score
            if region_score >= (0.87 * perfect_score):
                draw = ImageDraw.Draw(combine)
                top_left = (j,i)
                bottom_right = (j + t_width, i + t_height)
                #draw.rectangle(((100, 100), (200, 200)), (0, 255, 0))
                draw.rectangle((top_left, bottom_right), fill=None, outline = color)
                text_file_list.append([j, i, t_height, t_width, symbol_type, float(round((region_score/perfect_score*100), 2))])
#    combine.save("step5.png")
    return combine, text_file_list
    
def get_region_colors(im, t_height, t_width, coordinate):
    # coordinate is the x,y value of where the region starts in the image
    #print("im_height: " + str(im.height))
    #print("im_width: " + str(im.width))
    #print("get_region_colors: t_width " + str(t_width) + " t_height " + str(t_height))
    region_colors = []
    for i in range(coordinate[0], coordinate[0]+t_height):
        row = []
        for j in range(coordinate[1], coordinate[1]+t_width):
            #print(i, j)
            row.append(im.getpixel((j, i)))
        region_colors.append(row)
    return region_colors

def compareImages(region, template):
    # takes 2 matrices with the color values
    # region and template are the same size
    t_height = len(template)
    t_width = len(template[0])
    total_score = 0

    #print("compareImages: t_width" + str(t_width) + " t_height" + str(t_height))
    for i in range(t_height):
        for j in range(t_width):
            region_pixel = region[i][j]
            #print("region_pixel height" + str(len(region)))
            t_pixel = template[i][j]
            pixel_similarity = (region_pixel * t_pixel) + (255-region_pixel) * (255-t_pixel)
            total_score += pixel_similarity
    return total_score
```
## Hough Transform
The hough transform is used to detect lines in an image. We were able to visualized the hough space and calculate the rho and theta values but we weren't able to get the x coordinate from these values.

```
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
```
The goal of doing hough transform was to get the lines and space between the lines. We weren't exactly able to implement hough but we implemented a simple technique that achieves the same goal of the task. We first converted the image into a binary image and then calculated the number of pixels that were black for each row coordinate. After considering the row coordinates having the maximum values, we were able to find the space between the lines and get the location of the group of 5 lines present in the image. We further used these values to assign pitch values.

```
def hough(image):
    dist = 0
    img = image.convert('L')  
    # binary image
    bw = img.point(lambda x: 0 if x<128 else 255, '1')
    img_bin = np.array(bw).astype(np.uint8)
    x, y = img_bin.shape
    d = {}
    # count the number of black pixels for row coordinates
    for i in range(0,x):
        d[i] = 0
        for j in range(y):
            if img_bin[i][j]==0:
                d[i] +=1
    # consider keys which have values greater than half the width size
    l = [k for k,v in d.items() if v > y/2]
    
    #find the distance or space between the lines in a group
    for i in range(0,len(l)-1):
        if l[i]+1 != l[i+1]:
            if dist == 0:
                dist = l[i+1]-l[i]
            elif dist == l[i+1]-l[i]:
                break
    # get the group of lines positions        
    lines = [l[0]]
    p = l[0]
    for i in range(1,len(l)):
        if l[i] - p > dist*2:
            lines.append(l[i])
        p = l[i]
    
    return dist, lines
```

## Template matching using convolution

We implemented the Step 6 scoring function and dynamic programming to achieve better results but the running time was way too long due to O(n<sup>4</sup>). We tried to optimize it and could finally get it to run in a small amount of time. But we were not able to get the desired results. So, we went ahead with the template detection using a hamming distance.
The code for step-6 is as follows:-

``` 
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
```
    
The program does take a considerable amount of time to run and we would surely like to improve the run time in the future.