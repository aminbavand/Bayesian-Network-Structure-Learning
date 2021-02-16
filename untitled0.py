#Pa = []
#Pa.append([])#0
#Pa.append([2])#1
#Pa.append([])#2
#Pa.append([2])#3
#Pa.append([2])#4
#Pa.append([0,1,3])#5
#Pa.append([5])#6
#Pa.append([4])#7

#Pa = []
#Pa.append([])#0
#Pa.append([0])#1
#Pa.append([])#2
#Pa.append([2])#3
#Pa.append([2])#4
#Pa.append([1,3])#5
#Pa.append([5])#6
#Pa.append([4,5])#7

Pa = []
Pa.append([])#0
Pa.append([0])#1
Pa.append([])#2
Pa.append([2])#3
Pa.append([2])#4
Pa.append([1,3])#5
Pa.append([5])#6
Pa.append([4])#7


struct = tuple(tuple(x) for x in Pa)  
 
print(BIC_score(data,Pa,card,struct))