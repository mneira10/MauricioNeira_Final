import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.stats import norm

onlyfiles = [f for f in listdir(
    "./") if (isfile(join("./", f)) and f.endswith('.dat'))]


data = np.array([])

totData = []
fig = plt.figure(figsize=(12, 15))
for i, f in enumerate(onlyfiles):
    # print(f,i)
    data = np.loadtxt(f)
    totData.append(data)
    # data = np.concatenate([nums, data])
    plt.subplot(4, 2, i+1)
    plt.hist(data, normed=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel('Cadena {}'.format(i+1))

plt.savefig('punto1a.png')
plt.close()
# print(totData)
totData = np.array(totData)
# print(totData.shape)

N = 1000
M = len(onlyfiles)

def B(itera):
  totMean = totData[:,:itera].mean()
  ans = 0 
  for i in range(M):
    ans += (totData[i][:itera].mean()-totMean)**2
  return ans * N/(M-1)

def W(itera):
  ans = 0
  for i in range(M):
    ans += totData[i][:itera].var()
  return ans/M

def V(itera):
  return (N-1)*W(itera)/N + (M+1)*B(itera)/(M*N)

def gelmanRubin(itera):
  return V(itera)/W(itera)

# print(gelmanRubin(1000))
gr=[]
for i in range(1,1001):
  gr.append(gelmanRubin(i))

plt.plot(gr[1:])
plt.xlabel('# de iteraciones')
plt.ylabel('Estadística de Gelman-Rubin')
plt.savefig('gelmanRubin.png')
# plt.show()
# mu, std = norm.fit(data)

# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)
# # plt.show()
# plt.savefig('sample.pdf')
