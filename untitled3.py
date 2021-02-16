z= selected_pop1[0].copy()
for i in range(len(z)):
    z[i].append(i)
    
for i in range(len(z)-1):
    for j in range(i+1,len(z)):
        if len(set(z[i]) - (set(z[i]) - set(z[j])))>0:
            z[i] = list(set(z[i]+z[j]))
            z[j] = list(set(z[i]+z[j]))
s=10000000
for i in range(len(z)):
    if len(z[i])==len(z):
       s=0
print(s)