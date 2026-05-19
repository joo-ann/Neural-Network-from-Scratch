import matplotlib.pyplot as plt

primes = [2]

for i in range(3, 1000):
    prime = True
    for j in range(2, i):
        if i % j == 0:
            prime = False
    if prime:
        primes.append(i)

distances = []

for i in range(len(primes)-2):
    distances.append(primes[i+1]-primes[i])

plt.hist(distances)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Distances between Primes')
plt.show()
