import matplotlib.pyplot as plt

# Read data from timings.txt
n_values = []
cpu_times = []
gpu_times = []

with open("timings.txt", "r") as f:
    for line in f:
        n, cpu_time, gpu_time = map(float, line.split())
        n_values.append(int(n))
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

# Plot CPU and GPU times
plt.figure(figsize=(10, 6))
plt.plot(n_values, cpu_times, label='CPU Time (ms)', marker='o')
plt.plot(n_values, gpu_times, label='GPU Time (ms)', marker='o')

plt.xlabel('Array Size (n)')
plt.ylabel('Time (ms)')
plt.title('CPU vs GPU Time for Vector Addition')
plt.legend()
plt.grid(True)
plt.xscale('log')  # Use logarithmic scale for x-axis to handle large n
plt.show()
