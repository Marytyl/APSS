from numpy import unique

__GFEXP = [0] * 65536
	
# -- logarithms
__GFLOG = [0] * 65536
	
	
	
# prepare the exponential and logarithmic fields
__GFEXP[0] = 1
__GFLOG[0] = 'None'
byteValu = 1
for bytePos in range(1, 65535):
    byteValu <<= 1
    if (byteValu & 0x10000):
        byteValu ^= 0x1100b
			
		# update the field elements
    __GFEXP[bytePos] = byteValu
    __GFLOG[byteValu] = bytePos
		

print(len(__GFEXP))
print(len(__GFLOG))

print(len(unique(__GFEXP)))
print(len(unique(__GFLOG)))

file = open("gf216exp.txt", "w")
 
# Saving the array in a text file
content = str(__GFEXP)
file.write(content)
file.close()

file = open("gf216log.txt", "w")
 
# Saving the array in a text file
content = str(__GFLOG)
file.write(content)
file.close()



	
