import numpy as np
import matplotlib.pyplot as plt
import math


def convert_grayscale(image):
    '''  Convert image to grayscale image  '''
    
    #creating tempearray image
    grayScale = np.zeros( (image.shape[0], image.shape[1]) , dtype=np.uint8)
    
    #Taking each pixel and taking mean of RGB values
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grayScale[i,j] = int(image[i,j].mean())
        
    #grayScale,cmap = 'gray'
    return grayScale


def Convolution(image,kernal):
    '''Function Will perform Convolution'''
    # making our image ready for convolution
    
    #fliping kernal vertically and horizontaly
    kernal = np.flip(kernal)
    
    extra_units = math.floor(kernal.shape[0]/2)         # how many extra rows and coloumns we have to add
    
    
    conImage = np.zeros(shape=(image.shape[0]+(extra_units*2),image.shape[1]+(extra_units*2)))  # creating a new array of new size
    
    result = np.zeros(shape=(image.shape[0],image.shape[1]))  #final result
    
    
    #pasting image on ConImage 
    for i in range(image.shape[0]):    
        for j in range(image.shape[1]):
            conImage[i+extra_units][j+extra_units] = image[i][j]
    
    
    f_bit_value =0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            f_bit_value = 0
            #kernal multiplication
            for k_i in range(kernal.shape[0]):
                for k_j in range(kernal.shape[1]):
                    bit_value = kernal[k_i][k_j] * conImage[i+k_i][j+k_j]
                    f_bit_value = f_bit_value + bit_value
                    
            result[i][j] = f_bit_value
            
            
    return result


def noise_reduction(image):
    # applying guassin blur on grayscale image
    g_kernal = 1/16 * (np.array([[1,2,1],[2,4,2],[1,2,1]]))
    
    #COnvovling  gaussin kernal to image
    gBlurred_image = Convolution(image,g_kernal)
    
    return gBlurred_image



def gradient_calculation(image):
    #kernal for fx and fy
    fx_kernal = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    fy_kernal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    #Convolving fx with image 
    fx = Convolution(image,fx_kernal)
    plt.imsave("fx.jpg",fx,cmap = 'gray')

    #Convoving fy with image
    fy = Convolution(image,fy_kernal)
    plt.imsave("fy.jpg",fy,cmap = 'gray')
    
    
    #Creaing result image of same size as recvied image
    result = np.zeros(shape=(image.shape[0],image.shape[1]))
    
    # Now Calculating Gradient Magnitude
    # Farmula
    #  sqrt(fx^2 + fy^2)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i][j] = math.sqrt(fx[i][j]**2  + fy[i][j]**2)
            
    #finding gradient directions
    g_direction = gradient_direction(fx,fy)
            
    return result,g_direction

def gradient_direction(fx,fy):
    
    result = np.arctan2(fy,fx)
    
    #COnverting to degree
    result - np.rad2deg(result)
    
    #Converting to range 0-360
    result += 180
    
    return result


def non_maximum_suppression(GMag,GDir):
    '''Performs NMS, Takes 2 images Gradient Magnitude and Gradient Direction'''
    
    #Creating resultant Image
    NMS = np.zeros(GMag.shape)
    
    
    for i in range(1, GMag.shape[0] - 1):           #Not handling Boundary Cases
        for j in range(1, GMag.shape[1] - 1):
            
            if((GDir[i,j] >= 22.5 and GDir[i,j] <= 337.5) or (GDir[i,j] <= 157.5 and GDir[i,j] >= 202.5)):    #Horizantal Bin
                if((GMag[i,j] > GMag[i,j+1]) and (GMag[i,j] > GMag[i,j-1])):    #If neighboring pxls have grator magnitude than current pixel
                    NMS[i,j] = GMag[i,j]
                else:                                                           #else change the magnitude to 0
                    NMS[i,j] = 0         
            if((GDir[i,j] >= 337.5 and GDir[i,j] <= 292.5) or (GDir[i,j] <= 112.5 and GDir[i,j] >= 157.5)):   #Diagonal Bin
                if((GMag[i,j] > GMag[i+1,j+1]) and (GMag[i,j] > GMag[i-1,j-1])):
                    NMS[i,j] = GMag[i,j]
                else:
                    NMS[i,j] = 0
            if((GDir[i,j] >= 292.5 and GDir[i,j] <= 247.5) or (GDir[i,j] <= 67.5 and GDir[i,j] >= 112.5)):             #Vertical Bin
                if((GMag[i,j] > GMag[i+1,j]) and (GMag[i,j] > GMag[i-1,j])):
                    NMS[i,j] = GMag[i,j]
                else:
                    NMS[i,j] = 0
            if((GDir[i,j] >= 247.5 and GDir[i,j] <= 202.5) or (GDir[i,j] <= 22.5 and GDir[i,j] >= 67.5)):                #Diagonal Bin
                if((GMag[i,j] > GMag[i+1,j-1]) and (GMag[i,j] > GMag[i-1,j+1])):
                    NMS[i,j] = GMag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS




def double_threshold(NMS):
    '''Perfoems Double Thresholding take NMS created image'''
    
    #Constantly changed and check on which reslut is good
    THigh = 60
    
    TLow=THigh/3                                                        
    
    #Creating the resulting Image
    ThImg = np.zeros((NMS.shape[0],NMS.shape[1]))
    
    
    for i in range(NMS.shape[0]):
        for j in range(NMS.shape[1]):
            
            if(NMS[i,j]>=THigh):             #If True Make Strong Pixel -> 255
                ThImg[i,j]=255
            elif(NMS[i,j]<TLow):             #If true  make No-Edge Pixel -> 0    
                ThImg[i,j]=0
            else:            #else Make Weak Pixel  -> 50
                 ThImg[i,j]=50
                
    return ThImg



def hysteresis(ThImg):
    
    #Creating hysteresis image
    hysImg = np.zeros((ThImg.shape[0],ThImg.shape[1]))
        
    for i in range(ThImg.shape[0]):
        for j in range(ThImg.shape[1]):
             if(ThImg[i,j]>0 and ThImg[i,j]<=255):   #checking for weak pixel                         
                
                if((ThImg[i, j+1] == 255 or ThImg[i, j-1] == 255)   #Left and right neighnor
                   or
                   (ThImg[i+1, j-1] == 255 or ThImg[i-1, j+1] == 255)    #diagonal
                   or
                   (ThImg[i+1, j] == 255 or ThImg[i-1, j] == 255)     # Top Buttom
                   or
                   (ThImg[i+1, j+1] == 255 or ThImg[i-1, j-1] == 255)   #diagonal
                  ):                 
                    hysImg[i,j]=255
                else:
                    hysImg[i,j]=ThImg[i,j]
                
    
    return hysImg








def main():
    #   reading image
    image = plt.imread("book.jpg")

    #   Converting image to gray scale
    grayScale_image = convert_grayscale(image)

    #   Applying Gaussian Blur
    gBlurred_image = noise_reduction(grayScale_image)

    #   Applying Sobel Filter
    GMagnitude,GDirection = gradient_calculation(gBlurred_image)
    plt.imsave("GMagnitude.jpg", GMagnitude,cmap = 'gray')
    plt.imsave("GDirection.jpg", GDirection,cmap = 'gray')

    #   Applying Non-Maximum Suppression
    NMS = non_maximum_suppression(GMagnitude,GDirection)
    plt.imsave("NMS.jpg", NMS,cmap = 'gray')

    #   Applying Double Thresholding
    DThres = double_threshold(NMS)
    plt.imsave("DThres.jpg", DThres,cmap = 'gray')

    #   Applying Hysteresis
    hyp = hysteresis(DThres)
    plt.imsave("hyp.jpg", hyp,cmap = 'gray')




if __name__ == "__main__":
    main()








