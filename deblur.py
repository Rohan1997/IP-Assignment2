import numpy as np # Import numpy for mathematical operations and functions
import cv2
from matplotlib import pyplot as plt

def blur_1d(image,kernel):
	r,c = image.shape
	r1,c1 = kernel.shape
	rmax = max(r,r1)
	cmax = max(c,c1)
	pad_img = np.zeros((2*rmax,2*cmax))  
	pad_ker = np.zeros((2*rmax,2*cmax))

	kernel = kernel/np.sum(kernel)
	pad_img[0:r,0:c] = image
	pad_ker[0:r1,0:c1] = kernel

	fft_pad_img = np.fft.fft2(pad_img)
	fft_pad_ker = np.fft.fft2(pad_ker)

	fft_pad_img2 = np.fft.fftshift(fft_pad_img)
	fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)
	
	div = fft_pad_img2*(fft_pad_ker2)
	div2 = np.fft.ifftshift(div)
	div3 = np.fft.ifft2(div2).real

	divided = div3[0:r,0:c]
	return divided.astype(int)
	
def blur_3d(image,kernel):
	out_R = blur_1d(image[:,:,0],kernel[:,:,0])
	out_G = blur_1d(image[:,:,1],kernel[:,:,1])
	out_B = blur_1d(image[:,:,2],kernel[:,:,2])

	output = np.zeros((image.shape))
	output[:,:,0] = out_R
	output[:,:,1] = out_G
	output[:,:,2] = out_B

	return output.astype(int)
	
def deblur_1d(image,kernel):
	r,c = image.shape
	# kernel = cv2.resize(kernel,(r,c,))
	r1,c1 = kernel.shape
	rmax = max(r,r1)
	cmax = max(c,c1)
	pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
	pad_ker = np.zeros((2*rmax,2*cmax))

	kernel = kernel/np.sum(kernel)
	pad_img[0:r,0:c] = image
	pad_ker[0:r1,0:c1] = kernel

	fft_pad_img = np.fft.fft2(pad_img)
	fft_pad_ker = np.fft.fft2(pad_ker)

	fft_pad_img2 = np.fft.fftshift(fft_pad_img)
	fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)
	# print("min value =",np.min(np.abs(fft_pad_ker2)))
	# print("max value =",np.max(np.abs(fft_pad_ker2)))
	lim = 0.01
	print("lim =", lim)
	fft_pad_ker2[np.abs(fft_pad_ker2)<lim] = lim

	div = fft_pad_img2/(fft_pad_ker2)
	div2 = np.fft.ifftshift(div)
	div3 = np.fft.ifft2(div2).real

	divided = div3[0:r,0:c]
	# print("min value =",np.min(np.abs(divided)))
	# print("max value =",np.max(np.abs(divided)))
	# print("End")
	return divided.astype(int)

def deblur_3d(image,kernel):
	
	out_R = deblur_1d(image[:,:,0],kernel[:,:,0])
	out_G = deblur_1d(image[:,:,1],kernel[:,:,1])
	out_B = deblur_1d(image[:,:,2],kernel[:,:,2])

	output = np.zeros((image.shape))
	output[:,:,0] = out_R
	output[:,:,1] = out_G
	output[:,:,2] = out_B

	return output.astype(int)

def psnr(image, result):
    mse = np.mean( (image - result) ** 2 )
    if mse == 0:
    	return 100
    ans = 20*np.log10(255/np.sqrt(mse))
    return ans

def ssim(image,result):
	x = image
	y = result
	mean_x = np.mean(x)
	mean_y = np.mean(y)
	sigma_x = np.mean((x-mean_x)**2)
	sigma_y = np.mean((y-mean_y)**2)
	sigma_xy = np.mean((x-mean_x)*(y-mean_y)) 
	k1 = 0.01
	k2 = 0.03
	l =255
	c1 =k1*l
	c2 = k2*l
	ans = ( (2*mean_x*mean_y+c1)*(2*sigma_xy+c2) )/( (mean_x**2+mean_y**2+c1)*(sigma_x+sigma_y+c2) )
	# ans=0
	return ans

def wiener_filter1d(image, kernel):
	K1 = 0.01
	print("K =", K1)
	
	r,c = image.shape
	r1,c1 = kernel.shape
	rmax = max(r,r1)
	cmax = max(c,c1)
	pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
	pad_ker = np.zeros((2*rmax,2*cmax))

	kernel = kernel/np.sum(kernel)
	pad_img[0:r,0:c] = image
	pad_ker[0:r1,0:c1] = kernel

	fft_pad_img = np.fft.fft2(pad_img)
	fft_pad_ker = np.fft.fft2(pad_ker)

	fft_pad_img2 = np.fft.fftshift(fft_pad_img)
	fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)

	fft_pad_ker3 = (np.abs(fft_pad_ker2)**2) /( (fft_pad_ker2)*((np.abs(fft_pad_ker2)**2) + K1) )
	output = fft_pad_ker3*fft_pad_img2

	out = np.fft.ifftshift(output)
	out2 = np.fft.ifft2(out).real	
	out3 = out2[0:r,0:c]
	return out3.astype(int)

def wiener_filter3d(image,kernel):
	
	out_R = wiener_filter1d(image[:,:,0],kernel[:,:,0])
	out_G = wiener_filter1d(image[:,:,1],kernel[:,:,1])
	out_B = wiener_filter1d(image[:,:,2],kernel[:,:,2])

	output = np.zeros((image.shape))
	output[:,:,0] = out_R
	output[:,:,1] = out_G
	output[:,:,2] = out_B

	return output.astype(int)

def clsf1d(image,kernel):
	gamma1 = 0.00001
	print("Gamma =", gamma1)
	r,c = image.shape
	r1,c1 = kernel.shape
	rmax = max(r,r1)
	cmax = max(c,c1)
	p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	
	pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
	pad_ker = np.zeros((2*rmax,2*cmax))
	pad_p = np.zeros((2*rmax,2*cmax))
	
	kernel = kernel/np.sum(kernel)
	pad_img[0:r,0:c] = image
	pad_ker[0:r1,0:c1] = kernel
	pad_p[0:3,0:3] = p

	fft_pad_img = np.fft.fft2(pad_img)
	fft_pad_ker = np.fft.fft2(pad_ker)
	fft_pad_p = np.fft.fft2(pad_p)

	fft_pad_img2 = np.fft.fftshift(fft_pad_img)
	fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)
	fft_pad_p2 = np.fft.fftshift(fft_pad_p)

	fft_pad_ker3 = (np.conj(fft_pad_ker2)) /( (np.abs(fft_pad_ker2)**2) + gamma1*( np.abs(fft_pad_p2)**2 ) )
	
	output = fft_pad_ker3*fft_pad_img2

	out = np.fft.ifftshift(output)
	out2 = np.fft.ifft2(out).real
	
	out3 = out2[0:r,0:c]
	return out3.astype(int)

def clfs3d(image,kernel):
	
	out_R = clsf1d(image[:,:,0],kernel[:,:,0])
	out_G = clsf1d(image[:,:,1],kernel[:,:,1])
	out_B = clsf1d(image[:,:,2],kernel[:,:,2])

	output = np.zeros((image.shape))
	output[:,:,0] = out_R
	output[:,:,1] = out_G
	output[:,:,2] = out_B

	return output

def truncated_1d(image,kernel):
	r0 = 20
	print("r =", r0)
	# rad = r1
	r,c = image.shape
	r1,c1 = kernel.shape
	rmax = max(r,r1)
	cmax = max(c,c1)
	x0 = rmax
	y0 = cmax
	rx = (r0/100*x0)
	ry = (r0/100*y0)
	
	pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
	pad_ker = np.zeros((2*rmax,2*cmax))
	fft_trunc = np.ones((2*rmax,2*cmax),dtype=complex)
	
	kernel = kernel/np.sum(kernel)
	pad_img[0:r,0:c] = image
	pad_ker[0:r1,0:c1] = kernel

	fft_pad_img = np.fft.fft2(pad_img)
	fft_pad_ker = np.fft.fft2(pad_ker)

	fft_pad_img2 = np.fft.fftshift(fft_pad_img)
	fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)

	lim = 0.1
	# print("lim =", lim)
	fft_pad_ker2[np.abs(fft_pad_ker2)<lim] = lim

	fft_trunc[np.int_(x0-rx):np.int_(x0+rx),np.int_(y0-ry):np.int_(y0+ry)] = fft_pad_ker2[np.int_(x0-rx):np.int_(x0+rx),np.int_(y0-ry):np.int_(y0+ry)]
	
	div = fft_pad_img2/(fft_trunc)
	div2 = np.fft.ifftshift(div)
	div3 = np.fft.ifft2(div2).real

	divided = div3[0:r,0:c]
	
	return divided.astype(int)

def truncated_3d(image,kernel):
	
	out_R = truncated_1d(image[:,:,0],kernel[:,:,0])
	out_G = truncated_1d(image[:,:,1],kernel[:,:,1])
	out_B = truncated_1d(image[:,:,2],kernel[:,:,2])

	output = np.zeros((image.shape))
	output[:,:,0] = out_R
	output[:,:,1] = out_G
	output[:,:,2] = out_B

	return output.astype(int)

if __name__ == "__main__":
    
	img = plt.imread("/media/rohan/New Volume/IIT/SEM7/EE 610 Image Processing/Assignment2/IP_assignment2/Blurry4_1.png")
	filt = plt.imread("/media/rohan/New Volume/IIT/SEM7/EE 610 Image Processing/Assignment2/IP_assignment2/kernel1.png")
	truth = plt.imread("/media/rohan/New Volume/IIT/SEM7/EE 610 Image Processing/Assignment2/IP_assignment2/GroundTruth4_1_1.jpg")
	
	# img = cv2.imread("/media/rohan/New Volume/IIT/SEM7/EE 610 Image Processing/Assignment2/Blurry4_1.png")
	# filt = cv2.imread("/media/rohan/New Volume/IIT/SEM7/EE 610 Image Processing/Assignment2/kernel1.png")
	# truth = cv2.imread("/media/rohan/New Volume/IIT/SEM7/EE 610 Image Processing/Assignment2/GroundTruth4_1_1.jpg")
	
	img = 255*img
	filt = 255*filt
	truth = 255*truth

	if np.size(img.shape)==3:
		# final = blur_3d(img,filt)
		# final = deblur_3d(img,filt)
		# final = wiener_filter3d(img,filt)
		# final = clfs3d(img,filt)
		final = truncated_3d(img,filt)

	elif np.size(img.shape)==2:
		print(11)
		# final = deblur_1d(img,filt)
		# final = wiener_filter1d(img,filt)
		# final = clfs1d(img,filt)
		# final = truncated_1d(img,filt)
		
	else:
		print("Error1")
		final = np.zeros((img.shape))

	print("PSNR =",psnr(truth,final))
	print("SSIM =",ssim(truth,final))


	print(1)
	plt.figure()
	plt.imshow(np.int_(img))
	plt.title('Blurred image')
	
	final = np.int_(final)
	# final1 = final - np.min(final)
	final2 = np.int_(255*(final/np.max(final)))
	print("Max value of final =",np.max(final))
	
	plt.figure()
	plt.imshow(final)
	plt.title('Final image')
	# plt.savefig('trunc_50.png',bbox_inches='tight')
	
	
	# print(3)
	plt.figure()
	plt.imshow(np.int_(np.abs(final-img)) )	
	plt.title('Difference image')
	# plt.figure()
	
	# plt.imshow(truth)	
	# plt.title('Truth image')
	
	# cv2.imshow("hi",255*np.abs(fft_pad_ker2)/np.max(np.abs(fft_pad_ker2)))
	# cv2.waitKey(3000)
	# print(psnr(img,final))
	plt.show()